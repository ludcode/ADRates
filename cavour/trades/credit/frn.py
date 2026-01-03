##############################################################################

##############################################################################

"""
Floating Rate Note (FRN) implementation.

Provides the FRN class for creating, valuing, and analyzing floating-rate
bonds that pay coupons based on a reference rate (SONIA, SOFR, etc.) plus
a quoted margin (spread).

Key features:
- Floating rate coupon payments based on reference index
- Quoted margin (spread over reference rate)
- Optional caps and floors on coupon rates
- Clean and dirty price calculations
- Accrued interest computation
- Discount margin calculation
- Modified duration for floating rate instruments
- Integration with automatic differentiation for Greeks calculation

Example:
    >>> # Create a 5Y FRN paying SOFR + 50bp quarterly
    >>> value_dt = Date(15, 6, 2023)
    >>> frn = FRN(
    ...     issue_dt=Date(15, 6, 2023),
    ...     maturity_dt_or_tenor="5Y",
    ...     quoted_margin=0.005,  # 50bp spread
    ...     freq_type=FrequencyTypes.QUARTERLY,
    ...     dc_type=DayCountTypes.ACT_360,
    ...     currency=CurrencyTypes.USD,
    ...     floating_index=CurveTypes.USD_OIS_SOFR
    ... )
    >>>
    >>> # Value the FRN
    >>> pv = frn.value(value_dt, discount_curve, index_curve)
    >>> clean_px = frn.clean_price(value_dt, discount_curve, index_curve)
"""

import numpy as np
import jax.numpy as jnp
from scipy.optimize import newton, brentq

from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes, DayCount
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.calendar import CalendarTypes, DateGenRuleTypes
from cavour.utils.calendar import Calendar, BusDayAdjustTypes
from cavour.utils.schedule import Schedule
from cavour.utils.helpers import check_argument_types, label_to_string
from cavour.utils.global_types import InstrumentTypes, CurveTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.discount_curve import DiscountCurve

###############################################################################


class FRN:
    """
    Floating Rate Note (FRN) with coupons tied to a reference index.

    An FRN pays periodic coupons based on a floating reference rate
    (such as SONIA, SOFR, EURIBOR) plus a fixed quoted margin. The
    coupon rate resets periodically based on the prevailing reference rate.

    Pricing convention:
    - Dirty price = Present value including accrued interest
    - Clean price = Dirty price - Accrued interest
    - Prices quoted per 100 face value
    - Settlement typically T+1 or T+2 after trade date

    Supports:
    - Floating coupons with various frequencies and day count conventions
    - Quoted margin (fixed spread over reference rate)
    - Optional caps (maximum coupon rate)
    - Optional floors (minimum coupon rate)
    - Discount margin calculation
    - Modified duration calculation
    """

    def __init__(self,
                 issue_dt: Date,
                 maturity_dt_or_tenor: (Date, str),
                 quoted_margin: float,
                 freq_type: FrequencyTypes,
                 dc_type: DayCountTypes,
                 currency: CurrencyTypes,
                 floating_index: CurveTypes,
                 face_value: float = 100.0,
                 payment_lag: int = 0,
                 cap_rate: (float, type(None)) = None,
                 floor_rate: (float, type(None)) = None,
                 first_fixing_rate: (float, type(None)) = None,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING,
                 dg_type: DateGenRuleTypes = DateGenRuleTypes.BACKWARD,
                 end_of_month: bool = False):
        """
        Create a Floating Rate Note.

        Args:
            issue_dt: FRN issue date (when interest starts accruing)
            maturity_dt_or_tenor: Maturity date or tenor string (e.g., "5Y")
            quoted_margin: Fixed spread over reference rate (decimal, e.g., 0.005 for 50bp)
            freq_type: Coupon payment frequency (QUARTERLY, SEMI_ANNUAL, etc.)
            dc_type: Day count convention for accrual calculations
            currency: Currency of the FRN cashflows
            floating_index: Reference rate curve (e.g., USD_OIS_SOFR, GBP_OIS_SONIA)
            face_value: Face value (par) of the bond (default 100)
            payment_lag: Number of business days after period end for payment
            cap_rate: Optional maximum coupon rate (e.g., 0.06 for 6% cap)
            floor_rate: Optional minimum coupon rate (e.g., 0.0 for 0% floor)
            first_fixing_rate: Known first fixing rate for existing FRNs
            cal_type: Calendar for business day adjustments
            bd_type: Business day adjustment convention
            dg_type: Date generation rule (backward/forward from maturity)
            end_of_month: Whether to use end-of-month rule
        """

        check_argument_types(self.__init__, locals())

        # Store attributes
        self._issue_dt = issue_dt
        self._quoted_margin = quoted_margin
        self._freq_type = freq_type
        self._dc_type = dc_type
        self._currency = currency
        self._floating_index = floating_index
        self._face_value = face_value
        self._payment_lag = payment_lag
        self._cap_rate = cap_rate
        self._floor_rate = floor_rate
        self._first_fixing_rate = first_fixing_rate
        self._cal_type = cal_type
        self._bd_type = bd_type
        self._dg_type = dg_type
        self._end_of_month = end_of_month

        # Determine maturity date
        if isinstance(maturity_dt_or_tenor, Date):
            self._maturity_dt = maturity_dt_or_tenor
        else:
            self._maturity_dt = issue_dt.add_tenor(maturity_dt_or_tenor)

        calendar = Calendar(cal_type)
        self._maturity_dt = calendar.adjust(self._maturity_dt, bd_type)

        if issue_dt >= self._maturity_dt:
            raise LibError("Issue date must be before maturity date")

        # Initialize payment arrays
        self._payment_dts = []
        self._start_accrued_dts = []
        self._end_accrued_dts = []
        self._year_fracs = []
        self._accrued_days = []

        # Valuation arrays (populated by value() method)
        self._rates = []
        self._coupon_payments = []
        self._payment_dfs = []
        self._payment_pvs = []

        # Set derivative type for engine routing
        self.derivative_type = InstrumentTypes.FRN

        # Generate payment schedule
        self._generate_payment_schedule()

    ###########################################################################

    def _generate_payment_schedule(self):
        """
        Generate the coupon payment schedule using ISDA conventions.
        """

        schedule = Schedule(
            effective_dt=self._issue_dt,
            termination_dt=self._maturity_dt,
            freq_type=self._freq_type,
            cal_type=self._cal_type,
            bd_type=self._bd_type,
            dg_type=self._dg_type,
            end_of_month=self._end_of_month
        )

        schedule_dts = schedule._adjusted_dts

        if len(schedule_dts) < 2:
            raise LibError("Schedule must have at least two dates")

        self._payment_dts = []
        self._start_accrued_dts = []
        self._end_accrued_dts = []
        self._year_fracs = []
        self._accrued_days = []

        day_counter = DayCount(self._dc_type)
        calendar = Calendar(self._cal_type)

        prev_dt = schedule_dts[0]

        for next_dt in schedule_dts[1:]:
            self._start_accrued_dts.append(prev_dt)
            self._end_accrued_dts.append(next_dt)

            # Apply payment lag if specified
            if self._payment_lag == 0:
                payment_dt = next_dt
            else:
                payment_dt = calendar.add_business_days(next_dt, self._payment_lag)

            self._payment_dts.append(payment_dt)

            # Calculate year fraction and accrued days
            (year_frac, num_days, _) = day_counter.year_frac(prev_dt, next_dt)
            self._year_fracs.append(year_frac)
            self._accrued_days.append(num_days)

            prev_dt = next_dt

    ###########################################################################

    def value(self,
              value_dt: Date,
              discount_curve: DiscountCurve,
              index_curve: DiscountCurve = None,
              discount_margin: float = 0.0,
              settlement_dt: Date = None):
        """
        Calculate the present value of the FRN.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for PV calculation
            index_curve: Index curve for forward rate projection (if None, uses discount_curve)
            discount_margin: Additional spread for discounting (default 0)
            settlement_dt: Settlement date (if None, uses value_dt)

        Returns:
            Present value of the FRN in currency units (per face_value notional)
        """

        if discount_curve is None:
            raise LibError("Discount curve is required")

        if index_curve is None:
            index_curve = discount_curve

        if settlement_dt is None:
            settlement_dt = value_dt

        # Reset valuation arrays
        self._rates = []
        self._coupon_payments = []
        self._payment_dfs = []
        self._payment_pvs = []

        df_settle = discount_curve.df(settlement_dt, self._dc_type)
        pv = 0.0

        # Day count objects
        day_counter = DayCount(self._dc_type)
        index_dc = DayCount(index_curve._dc_type)

        first_payment = True

        # Value each coupon payment
        for i in range(len(self._payment_dts)):
            payment_dt = self._payment_dts[i]

            if payment_dt > settlement_dt:
                start_dt = self._start_accrued_dts[i]
                end_dt = self._end_accrued_dts[i]
                year_frac = self._year_fracs[i]

                # Calculate forward rate from index curve
                if first_payment and self._first_fixing_rate is not None:
                    # Use known first fixing
                    fwd_rate = self._first_fixing_rate
                    first_payment = False
                else:
                    # Calculate forward rate from index curve
                    (index_year_frac, _, _) = index_dc.year_frac(start_dt, end_dt)
                    df_start = index_curve.df(start_dt, self._dc_type)
                    df_end = index_curve.df(end_dt, self._dc_type)
                    fwd_rate = (df_start / df_end - 1.0) / index_year_frac

                # Apply quoted margin
                coupon_rate = fwd_rate + self._quoted_margin

                # Apply cap/floor if specified
                if self._cap_rate is not None:
                    coupon_rate = min(coupon_rate, self._cap_rate)
                if self._floor_rate is not None:
                    coupon_rate = max(coupon_rate, self._floor_rate)

                # Calculate coupon payment
                coupon_payment = coupon_rate * year_frac * self._face_value

                # Discount to settlement date (with optional discount margin)
                (disc_year_frac, _, _) = day_counter.year_frac(settlement_dt, payment_dt)
                df_payment = discount_curve.df(payment_dt, self._dc_type) / df_settle

                # Apply discount margin if specified
                if discount_margin != 0.0:
                    df_payment *= np.exp(-discount_margin * disc_year_frac)

                payment_pv = coupon_payment * df_payment
                pv += payment_pv

                # Store for later analysis
                self._rates.append(coupon_rate)
                self._coupon_payments.append(coupon_payment)
                self._payment_dfs.append(df_payment)
                self._payment_pvs.append(payment_pv)

            else:
                # Payment in the past
                self._rates.append(0.0)
                self._coupon_payments.append(0.0)
                self._payment_dfs.append(0.0)
                self._payment_pvs.append(0.0)

        # Add principal repayment at maturity
        if self._maturity_dt > settlement_dt:
            (disc_year_frac, _, _) = day_counter.year_frac(settlement_dt, self._maturity_dt)
            df_maturity = discount_curve.df(self._maturity_dt, self._dc_type) / df_settle

            # Apply discount margin if specified
            if discount_margin != 0.0:
                df_maturity *= np.exp(-discount_margin * disc_year_frac)

            principal_pv = self._face_value * df_maturity
            pv += principal_pv

            # Add to last payment for tracking
            if len(self._payment_pvs) > 0:
                self._payment_pvs[-1] += principal_pv

        return pv

    ###########################################################################

    def dirty_price(self,
                    value_dt: Date,
                    discount_curve: DiscountCurve,
                    index_curve: DiscountCurve = None,
                    discount_margin: float = 0.0,
                    settlement_dt: Date = None):
        """
        Calculate the dirty price (including accrued interest).

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for PV calculation
            index_curve: Index curve for forward rate projection
            discount_margin: Additional spread for discounting
            settlement_dt: Settlement date (if None, uses value_dt)

        Returns:
            Dirty price per 100 face value
        """

        pv = self.value(value_dt, discount_curve, index_curve, discount_margin, settlement_dt)
        return 100.0 * pv / self._face_value

    ###########################################################################

    def accrued_interest(self, settlement_dt: Date):
        """
        Calculate accrued interest from last coupon to settlement.

        For FRNs, accrued interest is typically calculated using the previous
        fixing rate plus the quoted margin. If no fixing is available, uses
        an estimate.

        Args:
            settlement_dt: Settlement date for accrual calculation

        Returns:
            Accrued interest per 100 face value
        """

        day_counter = DayCount(self._dc_type)

        # Find the relevant coupon period
        for i in range(len(self._payment_dts)):
            if self._payment_dts[i] > settlement_dt:
                start_dt = self._start_accrued_dts[i]
                end_dt = self._end_accrued_dts[i]

                if settlement_dt >= start_dt:
                    # Calculate accrued fraction
                    (accrued_frac, _, _) = day_counter.year_frac(start_dt, settlement_dt)
                    (period_frac, _, _) = day_counter.year_frac(start_dt, end_dt)

                    # Use first fixing if available, otherwise estimate
                    if self._first_fixing_rate is not None:
                        accrual_rate = self._first_fixing_rate + self._quoted_margin
                    else:
                        # Estimate using quoted margin only (conservative)
                        accrual_rate = self._quoted_margin

                    # Apply cap/floor if specified
                    if self._cap_rate is not None:
                        accrual_rate = min(accrual_rate, self._cap_rate)
                    if self._floor_rate is not None:
                        accrual_rate = max(accrual_rate, self._floor_rate)

                    accrued = accrual_rate * accrued_frac * self._face_value
                    return 100.0 * accrued / self._face_value

        # No accrual if before first period or after maturity
        return 0.0

    ###########################################################################

    def clean_price(self,
                    value_dt: Date,
                    discount_curve: DiscountCurve,
                    index_curve: DiscountCurve = None,
                    discount_margin: float = 0.0,
                    settlement_dt: Date = None):
        """
        Calculate the clean price (dirty price minus accrued interest).

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for PV calculation
            index_curve: Index curve for forward rate projection
            discount_margin: Additional spread for discounting
            settlement_dt: Settlement date (if None, uses value_dt)

        Returns:
            Clean price per 100 face value
        """

        if settlement_dt is None:
            settlement_dt = value_dt

        dirty = self.dirty_price(value_dt, discount_curve, index_curve, discount_margin, settlement_dt)
        accrued = self.accrued_interest(settlement_dt)
        return dirty - accrued

    ###########################################################################

    def discount_margin(self,
                       settlement_dt: Date,
                       discount_curve: DiscountCurve,
                       index_curve: DiscountCurve,
                       clean_price: float,
                       dm_guess: float = 0.0):
        """
        Calculate the discount margin that produces the given clean price.

        The discount margin (DM) is the parallel spread over the discount curve
        that equates the FRN's price to the market price. It's analogous to
        YTM for fixed-rate bonds.

        Args:
            settlement_dt: Settlement date
            discount_curve: Discount curve for PV calculation
            index_curve: Index curve for forward rate projection
            clean_price: Target clean price per 100 face value
            dm_guess: Initial guess for discount margin (default 0.0)

        Returns:
            Discount margin (as decimal, e.g., 0.01 for 100bp)
        """

        accrued = self.accrued_interest(settlement_dt)
        target_dirty = clean_price + accrued

        def price_error(dm):
            calc_dirty = self.dirty_price(settlement_dt, discount_curve, index_curve, dm, settlement_dt)
            return calc_dirty - target_dirty

        try:
            # Try Brent's method first (more robust)
            dm = brentq(price_error, -0.10, 0.20, xtol=1e-8)
        except:
            # Fall back to Newton-Raphson
            try:
                dm = newton(price_error, dm_guess, tol=1e-8, maxiter=50)
            except:
                raise LibError(f"Failed to converge on discount margin for price {clean_price}")

        return dm

    ###########################################################################

    def modified_duration(self,
                         value_dt: Date,
                         discount_curve: DiscountCurve,
                         index_curve: DiscountCurve = None,
                         discount_margin: float = 0.0,
                         settlement_dt: Date = None):
        """
        Calculate modified duration for the FRN.

        FRNs typically have low duration because coupon rates reset periodically.
        Duration is approximately equal to the time until the next reset date.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve
            index_curve: Index curve for forward rate projection
            discount_margin: Discount margin for calculation
            settlement_dt: Settlement date

        Returns:
            Modified duration in years
        """

        if settlement_dt is None:
            settlement_dt = value_dt

        # Calculate dirty price
        p0 = self.dirty_price(value_dt, discount_curve, index_curve, discount_margin, settlement_dt)

        # Bump all curve rates by 1bp
        bump = 0.0001

        # Create bumped curve (simplified - full implementation would bump pillar rates)
        # For now, approximate using discount margin bump
        p_up = self.dirty_price(value_dt, discount_curve, index_curve, discount_margin + bump, settlement_dt)
        p_down = self.dirty_price(value_dt, discount_curve, index_curve, discount_margin - bump, settlement_dt)

        # Numerical derivative: duration = -(1/P)(dP/dy)
        duration = -(p_up - p_down) / (2 * bump * p0)

        return duration

    ###########################################################################

    def dv01(self,
             value_dt: Date,
             discount_curve: DiscountCurve,
             index_curve: DiscountCurve = None,
             discount_margin: float = 0.0,
             settlement_dt: Date = None):
        """
        Calculate DV01 (dollar value of 1 basis point) for the FRN.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve
            index_curve: Index curve
            discount_margin: Discount margin
            settlement_dt: Settlement date

        Returns:
            DV01 (change in value for 1bp rate change)
        """

        if settlement_dt is None:
            settlement_dt = value_dt

        # Calculate PV
        pv = self.value(value_dt, discount_curve, index_curve, discount_margin, settlement_dt)

        # Bump discount margin by 1bp
        bump = 0.0001
        pv_bumped = self.value(value_dt, discount_curve, index_curve, discount_margin + bump, settlement_dt)

        # DV01 = change in value for 1bp
        dv01 = abs(pv_bumped - pv)

        return dv01

    ###########################################################################

    def position(self, model):
        """
        Create a Position object for use with automatic differentiation.

        Args:
            model: Model containing market data and curves

        Returns:
            Position object for AD calculations
        """

        # Import locally to avoid circular import
        from cavour.market.position.position import Position
        return Position(self, model)

    ###########################################################################

    def print_payments(self):
        """
        Print the payment schedule (without valuations).
        """

        print("=" * 80)
        print("FRN PAYMENT SCHEDULE")
        print("=" * 80)
        print(f"Issue Date:        {self._issue_dt}")
        print(f"Maturity Date:     {self._maturity_dt}")
        print(f"Quoted Margin:     {self._quoted_margin * 10000:.2f} bp")
        print(f"Frequency:         {self._freq_type}")
        print(f"Day Count:         {self._dc_type}")
        print(f"Face Value:        {self._face_value:.2f}")
        print(f"Currency:          {self._currency}")
        print(f"Floating Index:    {self._floating_index}")
        if self._cap_rate is not None:
            print(f"Cap Rate:          {self._cap_rate * 100:.4f}%")
        if self._floor_rate is not None:
            print(f"Floor Rate:        {self._floor_rate * 100:.4f}%")
        print("=" * 80)

        print(f"\n{'Num':<5} {'Pay Date':<12} {'Start':<12} {'End':<12} {'Days':<6} {'Year Frac':<10}")
        print("-" * 80)

        for i in range(len(self._payment_dts)):
            print(f"{i+1:<5} "
                  f"{str(self._payment_dts[i]):<12} "
                  f"{str(self._start_accrued_dts[i]):<12} "
                  f"{str(self._end_accrued_dts[i]):<12} "
                  f"{self._accrued_days[i]:<6} "
                  f"{self._year_fracs[i]:<10.6f}")

    ###########################################################################

    def print_valuation(self):
        """
        Print the full valuation (requires value() to have been called first).
        """

        if not self._rates:
            print("No valuation available. Call value() first.")
            return

        print("=" * 80)
        print("FRN VALUATION")
        print("=" * 80)

        total_pv = sum(self._payment_pvs)

        print(f"\n{'Num':<5} {'Pay Date':<12} {'Rate %':<10} {'Payment':<15} {'DF':<10} {'PV':<15}")
        print("-" * 80)

        for i in range(len(self._payment_dts)):
            print(f"{i+1:<5} "
                  f"{str(self._payment_dts[i]):<12} "
                  f"{self._rates[i] * 100:<10.4f} "
                  f"{self._coupon_payments[i]:<15.2f} "
                  f"{self._payment_dfs[i]:<10.6f} "
                  f"{self._payment_pvs[i]:<15.2f}")

        print("-" * 80)
        print(f"{'Total PV:':<50} {total_pv:<15.2f}")
        print("=" * 80)

    ###########################################################################

    def __repr__(self):
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("ISSUE DATE", self._issue_dt)
        s += label_to_string("MATURITY DATE", self._maturity_dt)
        s += label_to_string("QUOTED MARGIN (BP)", self._quoted_margin * 10000)
        s += label_to_string("FREQUENCY", self._freq_type)
        s += label_to_string("DAY COUNT", self._dc_type)
        s += label_to_string("CURRENCY", self._currency)
        s += label_to_string("FLOATING INDEX", self._floating_index)
        s += label_to_string("FACE VALUE", self._face_value)
        if self._cap_rate is not None:
            s += label_to_string("CAP RATE (%)", self._cap_rate * 100)
        if self._floor_rate is not None:
            s += label_to_string("FLOOR RATE (%)", self._floor_rate * 100)
        return s

###############################################################################
