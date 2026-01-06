##############################################################################

##############################################################################

"""
Bond implementation for fixed-coupon and zero-coupon bonds.

Provides the Bond class for creating, valuing, and analyzing fixed income
securities including vanilla fixed-coupon bonds and zero-coupon bonds.

Key features:
- Fixed coupon bond construction using ISDA conventions
- Zero coupon bond support
- Clean and dirty price calculations
- Accrued interest computation
- Yield to maturity (YTM) calculation
- Spread analysis (Z-spread, G-spread, I-spread)
- Risk measures (duration, convexity, DV01)
- Integration with automatic differentiation for Greeks calculation

Example:
    >>> # Create a 10Y 5% annual coupon bond
    >>> value_dt = Date(15, 6, 2023)
    >>> bond = Bond(
    ...     issue_dt=Date(15, 6, 2023),
    ...     maturity_dt_or_tenor="10Y",
    ...     coupon=0.05,
    ...     freq_type=FrequencyTypes.ANNUAL,
    ...     dc_type=DayCountTypes.ACT_365F,
    ...     currency=CurrencyTypes.GBP
    ... )
    >>>
    >>> # Value the bond
    >>> pv = bond.value(value_dt, discount_curve)
    >>> clean_px = bond.clean_price(value_dt, discount_curve)
    >>> ytm = bond.yield_to_maturity(value_dt, clean_px)
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
from cavour.utils.global_types import InstrumentTypes
from cavour.utils.currency import CurrencyTypes
from cavour.market.curves.discount_curve import DiscountCurve

###############################################################################


class Bond:
    """
    Fixed or zero coupon bond with regular coupon payments.

    A bond is a debt instrument where the issuer pays regular coupons
    (interest payments) and returns the principal (face value) at maturity.
    Bonds are priced using a discount curve, optionally with a Z-spread
    adjustment for credit risk.

    Pricing convention:
    - Dirty price = Present value including accrued interest
    - Clean price = Dirty price - Accrued interest
    - Prices quoted per 100 face value
    - Settlement typically T+1 or T+2 after trade date

    Supports:
    - Fixed coupon bonds with various frequencies and day count conventions
    - Zero coupon bonds (coupon=0 or freq_type=ZERO)
    - Z-spread bumping for credit sensitivity
    - Settlement date conventions
    """

    def __init__(self,
                 issue_dt: Date,
                 maturity_dt_or_tenor: (Date, str),
                 coupon: float,
                 freq_type: FrequencyTypes,
                 dc_type: DayCountTypes,
                 currency: CurrencyTypes,
                 face_value: float = 100.0,
                 payment_lag: int = 0,
                 amortization_schedule: (list, type(None)) = None,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING,
                 dg_type: DateGenRuleTypes = DateGenRuleTypes.BACKWARD,
                 end_of_month: bool = False):
        """
        Create a fixed or zero coupon bond (with optional amortization).

        Args:
            issue_dt: Bond issue date (when interest starts accruing)
            maturity_dt_or_tenor: Maturity date or tenor string (e.g., "10Y")
            coupon: Annual coupon rate (decimal, e.g., 0.05 for 5%)
            freq_type: Coupon payment frequency (ANNUAL, SEMI_ANNUAL, etc.)
            dc_type: Day count convention (ACT_365F, ACT_ACT_ISDA, 30/360, etc.)
            currency: Currency of the bond
            face_value: Face value / par value (default 100)
            payment_lag: Days between accrual end and payment (default 0)
            amortization_schedule: Optional list of outstanding principal at each
                payment date. If None, bullet bond (full principal at maturity).
                If provided, must match number of coupon periods.
                Example: [80, 60, 40, 20, 0] for 5-period equal principal amortization.
            cal_type: Holiday calendar for payment dates
            bd_type: Business day adjustment convention
            dg_type: Date generation rule (BACKWARD or FORWARD)
            end_of_month: End of month adjustment flag
        """
        check_argument_types(self.__init__, locals())

        self.derivative_type = InstrumentTypes.BOND

        # Determine maturity date
        if isinstance(maturity_dt_or_tenor, Date):
            self._maturity_dt = maturity_dt_or_tenor
        else:
            self._maturity_dt = issue_dt.add_tenor(maturity_dt_or_tenor)

        # Validate dates
        if issue_dt >= self._maturity_dt:
            raise LibError("Issue date must be before maturity date")

        # Store bond parameters
        self._issue_dt = issue_dt
        self._coupon = coupon
        self._freq_type = freq_type
        self._dc_type = dc_type
        self._currency = currency
        self._face_value = face_value
        self._payment_lag = payment_lag
        self._cal_type = cal_type
        self._bd_type = bd_type
        self._dg_type = dg_type
        self._end_of_month = end_of_month
        self._amortization_schedule = amortization_schedule

        # Generate payment schedule
        self._is_zero_coupon = (coupon == 0.0 or freq_type == FrequencyTypes.ZERO)

        if not self._is_zero_coupon:
            self._generate_coupon_schedule()
        else:
            # Zero coupon bond - only principal at maturity
            self._payment_dts = [self._maturity_dt]
            self._year_fracs = [0.0]
            self._coupon_payments = [0.0]
            self._accrual_start_dts = [issue_dt]
            self._accrual_end_dts = [self._maturity_dt]
            self._num_coupons = 0
            # Initialize principal schedule for zero-coupon bonds
            self._principal_schedule = [self._face_value, 0.0]
            self._principal_payments = [self._face_value]

###############################################################################

    def _generate_coupon_schedule(self):
        """
        Generate the coupon payment schedule using ISDA conventions.

        Creates payment dates, calculates year fractions, and computes
        coupon amounts for each payment period. For amortizing bonds,
        coupons are calculated on declining principal balance.
        """
        # Create calendar for business day adjustments
        calendar = Calendar(self._cal_type)

        # Generate schedule using ISDA conventions
        schedule = Schedule(
            effective_dt=self._issue_dt,
            termination_dt=self._maturity_dt,
            freq_type=self._freq_type,
            cal_type=self._cal_type,
            bd_type=self._bd_type,
            dg_type=self._dg_type,
            end_of_month=self._end_of_month
        )

        # Get schedule dates
        schedule_dts = schedule._adjusted_dts

        # Initialize storage
        self._accrual_start_dts = []
        self._accrual_end_dts = []
        self._payment_dts = []
        self._year_fracs = []
        self._coupon_payments = []
        self._principal_schedule = []
        self._principal_payments = []

        # Validate amortization schedule if provided
        num_periods = len(schedule_dts) - 1
        if self._amortization_schedule is not None:
            if len(self._amortization_schedule) != num_periods:
                raise LibError(
                    f"Amortization schedule length ({len(self._amortization_schedule)}) "
                    f"must match number of payment periods ({num_periods})"
                )

            # Create full schedule with initial face value
            self._principal_schedule = [self._face_value] + list(self._amortization_schedule)
        else:
            # Bullet bond - constant principal until maturity, then 0
            self._principal_schedule = [self._face_value] * num_periods + [0.0]

        # Calculate year fractions and payments for each period
        day_count = DayCount(self._dc_type)
        prev_dt = self._issue_dt

        for i, next_dt in enumerate(schedule_dts[1:]):
            # Accrual period
            accrual_start = prev_dt
            accrual_end = next_dt

            # Payment date (with lag if specified)
            payment_dt = calendar.add_business_days(accrual_end, self._payment_lag)

            # Calculate year fraction for coupon accrual
            year_frac = day_count.year_frac(accrual_start, accrual_end)[0]

            # Outstanding principal at start of period (for coupon calculation)
            outstanding_principal = self._principal_schedule[i]

            # Calculate coupon payment on outstanding principal
            coupon_payment = year_frac * self._coupon * outstanding_principal

            # Calculate principal repayment this period
            principal_payment = self._principal_schedule[i] - self._principal_schedule[i + 1]

            # Store
            self._accrual_start_dts.append(accrual_start)
            self._accrual_end_dts.append(accrual_end)
            self._payment_dts.append(payment_dt)
            self._year_fracs.append(year_frac)
            self._coupon_payments.append(coupon_payment)
            self._principal_payments.append(principal_payment)

            prev_dt = next_dt

        self._num_coupons = len(self._payment_dts)

###############################################################################

    def position(self, model):
        """
        Create a Position object for this bond.

        Args:
            model: Model object containing curves and market data

        Returns:
            Position object for computing risk measures
        """
        from cavour.market.position.position import Position
        return Position(self, model)

###############################################################################

    def value(self,
              value_dt: Date,
              discount_curve: DiscountCurve,
              z_spread: float = 0.0,
              settlement_dt: Date = None):
        """
        Calculate the present value of the bond.

        Values all future coupon payments and principal using the discount
        curve, with optional Z-spread adjustment for credit risk.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve for PV calculation
            z_spread: Z-spread in decimal (e.g., 0.01 for 100bp)
            settlement_dt: Settlement date (defaults to value_dt for T+0)

        Returns:
            Present value of the bond in currency units
        """
        if settlement_dt is None:
            settlement_dt = value_dt

        # Get discount factor at settlement date
        df_settlement = discount_curve.df(settlement_dt)

        # Initialize cashflow tracking arrays (for cashflow extraction)
        self._payment_dfs = []
        self._coupon_pvs = []
        self._principal_pvs = []

        bond_pv = 0.0

        # Value coupon payments
        for i, payment_dt in enumerate(self._payment_dts):
            if payment_dt > settlement_dt:
                # Get discount factor at payment date
                df_payment = discount_curve.df(payment_dt)

                # Apply Z-spread adjustment if specified
                if z_spread != 0.0:
                    time_to_payment = (payment_dt - settlement_dt) / 365.25
                    df_payment = df_payment * np.exp(-z_spread * time_to_payment)

                # Relative discount factor
                df_payment_rel = df_payment / df_settlement

                # PV of coupon
                coupon_pv = self._coupon_payments[i] * df_payment_rel
                bond_pv += coupon_pv

                # Store for cashflow extraction
                self._payment_dfs.append(df_payment_rel)
                self._coupon_pvs.append(coupon_pv)
            else:
                # Past payment - store zeros
                self._payment_dfs.append(0.0)
                self._coupon_pvs.append(0.0)

        # Value principal repayments (for amortizing bonds, these occur at each payment date)
        if hasattr(self, '_principal_payments'):
            # Amortizing bond - principal payments at each coupon date
            for i, payment_dt in enumerate(self._payment_dts):
                if payment_dt > settlement_dt and self._principal_payments[i] > 0:
                    # Get discount factor at payment date
                    df_payment = discount_curve.df(payment_dt)

                    # Apply Z-spread adjustment
                    if z_spread != 0.0:
                        time_to_payment = (payment_dt - settlement_dt) / 365.25
                        df_payment = df_payment * np.exp(-z_spread * time_to_payment)

                    df_payment_rel = df_payment / df_settlement
                    principal_pv = self._principal_payments[i] * df_payment_rel
                    bond_pv += principal_pv

                    # Store for cashflow extraction
                    self._principal_pvs.append(principal_pv)
                else:
                    self._principal_pvs.append(0.0)
        else:
            # Bullet bond - all principal at maturity
            # Initialize principal PVs for all payment dates as zero
            self._principal_pvs = [0.0] * len(self._payment_dts)

            if self._maturity_dt > settlement_dt:
                df_maturity = discount_curve.df(self._maturity_dt)

                # Apply Z-spread adjustment
                if z_spread != 0.0:
                    time_to_maturity = (self._maturity_dt - settlement_dt) / 365.25
                    df_maturity = df_maturity * np.exp(-z_spread * time_to_maturity)

                df_maturity_rel = df_maturity / df_settlement
                principal_pv = self._face_value * df_maturity_rel
                bond_pv += principal_pv

                # Store principal PV at maturity (last payment date)
                self._principal_pvs[-1] = principal_pv

        return bond_pv

###############################################################################

    def accrued_interest(self, settlement_dt: Date):
        """
        Calculate accrued interest from last coupon to settlement date.

        Args:
            settlement_dt: Settlement date for accrued calculation

        Returns:
            Accrued interest amount
        """
        if self._is_zero_coupon:
            return 0.0

        # Find the last coupon date on or before settlement
        last_coupon_dt = self._issue_dt
        next_coupon_dt = self._maturity_dt
        year_frac_period = 1.0

        for i, payment_dt in enumerate(self._payment_dts):
            if payment_dt <= settlement_dt:
                last_coupon_dt = self._accrual_end_dts[i]
            elif payment_dt > settlement_dt:
                next_coupon_dt = self._accrual_end_dts[i]
                last_coupon_dt = self._accrual_start_dts[i]
                year_frac_period = self._year_fracs[i]
                break

        # Calculate accrued interest
        day_count = DayCount(self._dc_type)
        accrued_year_frac = day_count.year_frac(last_coupon_dt, settlement_dt)[0]
        accrued_interest = accrued_year_frac * self._coupon * self._face_value

        return accrued_interest

###############################################################################

    def dirty_price(self,
                     value_dt: Date,
                     discount_curve: DiscountCurve,
                     z_spread: float = 0.0,
                     settlement_dt: Date = None):
        """
        Calculate dirty price (full price including accrued interest).

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve
            z_spread: Z-spread in decimal
            settlement_dt: Settlement date (defaults to value_dt)

        Returns:
            Dirty price per 100 face value
        """
        if settlement_dt is None:
            settlement_dt = value_dt

        pv = self.value(value_dt, discount_curve, z_spread, settlement_dt)
        dirty_px = (pv / self._face_value) * 100.0

        return dirty_px

###############################################################################

    def clean_price(self,
                     value_dt: Date,
                     discount_curve: DiscountCurve,
                     z_spread: float = 0.0,
                     settlement_dt: Date = None):
        """
        Calculate clean price (price without accrued interest).

        This is the quoted price in bond markets.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve
            z_spread: Z-spread in decimal
            settlement_dt: Settlement date (defaults to value_dt)

        Returns:
            Clean price per 100 face value
        """
        if settlement_dt is None:
            settlement_dt = value_dt

        dirty_px = self.dirty_price(value_dt, discount_curve, z_spread, settlement_dt)
        accrued = self.accrued_interest(settlement_dt)
        accrued_per_100 = (accrued / self._face_value) * 100.0

        clean_px = dirty_px - accrued_per_100

        return clean_px

###############################################################################

    def yield_to_maturity(self,
                          settlement_dt: Date,
                          clean_price: float):
        """
        Calculate yield to maturity given a clean price.

        Solves for the discount rate that equates the present value of
        all cashflows to the given clean price.

        Args:
            settlement_dt: Settlement date
            clean_price: Clean price per 100 face value

        Returns:
            Yield to maturity (annualized, as decimal)
        """
        # Convert clean price to dirty price
        accrued = self.accrued_interest(settlement_dt)
        accrued_per_100 = (accrued / self._face_value) * 100.0
        dirty_price = clean_price + accrued_per_100

        # Target PV
        target_pv = (dirty_price / 100.0) * self._face_value

        # Define function to solve: PV(ytm) - target_pv = 0
        def pv_difference(ytm):
            pv = 0.0

            # Value coupons
            for i, payment_dt in enumerate(self._payment_dts):
                if payment_dt > settlement_dt:
                    time_to_payment = (payment_dt - settlement_dt) / 365.25
                    df = np.exp(-ytm * time_to_payment)
                    pv += self._coupon_payments[i] * df

            # Value principal
            if self._maturity_dt > settlement_dt:
                time_to_maturity = (self._maturity_dt - settlement_dt) / 365.25
                df = np.exp(-ytm * time_to_maturity)
                pv += self._face_value * df

            return pv - target_pv

        # Solve using Brent's method (robust for YTM calculation)
        try:
            ytm = brentq(pv_difference, -0.5, 0.5, maxiter=100)
        except:
            # Fallback to Newton's method if Brent fails
            ytm = newton(pv_difference, 0.05, maxiter=100)

        return ytm

###############################################################################

    def current_yield(self):
        """
        Calculate current yield (annual coupon / price).

        Note: This is a simplified yield measure that doesn't account for
        time value of money or capital gains/losses.

        Returns:
            Current yield as decimal
        """
        if self._is_zero_coupon:
            return 0.0

        return self._coupon

###############################################################################

    def z_spread(self,
                 settlement_dt: Date,
                 discount_curve: DiscountCurve,
                 clean_price: float):
        """
        Calculate Z-spread given a clean price.

        Solves for the parallel spread over the discount curve that equates
        the present value to the given clean price.

        Args:
            settlement_dt: Settlement date
            discount_curve: Base discount curve
            clean_price: Clean price per 100 face value

        Returns:
            Z-spread as decimal (e.g., 0.01 for 100bp)
        """
        # Convert clean price to dirty price
        accrued = self.accrued_interest(settlement_dt)
        accrued_per_100 = (accrued / self._face_value) * 100.0
        dirty_price = clean_price + accrued_per_100

        # Target PV
        target_pv = (dirty_price / 100.0) * self._face_value

        # Define function to solve: PV(z_spread) - target_pv = 0
        def pv_difference(z_spr):
            pv = self.value(settlement_dt, discount_curve, z_spr, settlement_dt)
            return pv - target_pv

        # Solve using Brent's method
        try:
            z_spr = brentq(pv_difference, -0.1, 0.5, maxiter=100)
        except:
            # Fallback to Newton's method
            z_spr = newton(pv_difference, 0.01, maxiter=100)

        return z_spr

###############################################################################

    def g_spread(self,
                 settlement_dt: Date,
                 govt_curve: DiscountCurve,
                 clean_price: float):
        """
        Calculate G-spread (spread over government curve).

        This is simply the difference between the bond's YTM and the
        government curve's yield at the bond's maturity.

        Args:
            settlement_dt: Settlement date
            govt_curve: Government discount curve
            clean_price: Clean price per 100 face value

        Returns:
            G-spread as decimal (e.g., 0.01 for 100bp)
        """
        # Calculate bond's YTM
        bond_ytm = self.yield_to_maturity(settlement_dt, clean_price)

        # Get government curve yield at maturity
        from cavour.utils.frequency import annual_frequency
        freq = annual_frequency(self._freq_type)
        govt_yield = govt_curve.zero_rate(
            self._maturity_dt,
            freq_type=self._freq_type,
            dc_type=self._dc_type
        )

        # G-spread is the difference
        g_spr = bond_ytm - govt_yield

        return g_spr

###############################################################################

    def i_spread(self,
                 settlement_dt: Date,
                 discount_curve: DiscountCurve,
                 clean_price: float):
        """
        Calculate I-spread (interpolated spread).

        This is the difference between the bond's YTM and the swap curve's
        interpolated yield at the bond's maturity.

        Args:
            settlement_dt: Settlement date
            discount_curve: Swap/OIS discount curve
            clean_price: Clean price per 100 face value

        Returns:
            I-spread as decimal (e.g., 0.01 for 100bp)
        """
        # Calculate bond's YTM
        bond_ytm = self.yield_to_maturity(settlement_dt, clean_price)

        # Get swap curve yield at maturity
        swap_yield = discount_curve.zero_rate(
            self._maturity_dt,
            freq_type=self._freq_type,
            dc_type=self._dc_type
        )

        # I-spread is the difference
        i_spr = bond_ytm - swap_yield

        return i_spr

###############################################################################

    def duration(self,
                 settlement_dt: Date,
                 discount_curve: DiscountCurve,
                 duration_type: str = 'modified',
                 z_spread: float = 0.0):
        """
        Calculate bond duration (Macaulay or Modified).

        Duration measures the price sensitivity to yield changes.

        Args:
            settlement_dt: Settlement date
            discount_curve: Discount curve
            duration_type: 'macaulay' or 'modified'
            z_spread: Z-spread in decimal

        Returns:
            Duration in years
        """
        # Get current price and YTM
        clean_px = self.clean_price(settlement_dt, discount_curve, z_spread, settlement_dt)
        ytm = self.yield_to_maturity(settlement_dt, clean_px)

        # Calculate Macaulay duration
        weighted_time = 0.0
        total_pv = 0.0

        for i, payment_dt in enumerate(self._payment_dts):
            if payment_dt > settlement_dt:
                time_to_payment = (payment_dt - settlement_dt) / 365.25
                df = np.exp(-ytm * time_to_payment)
                pv = self._coupon_payments[i] * df
                weighted_time += pv * time_to_payment
                total_pv += pv

        # Principal
        if self._maturity_dt > settlement_dt:
            time_to_maturity = (self._maturity_dt - settlement_dt) / 365.25
            df = np.exp(-ytm * time_to_maturity)
            pv = self._face_value * df
            weighted_time += pv * time_to_maturity
            total_pv += pv

        macaulay_duration = weighted_time / total_pv

        if duration_type.lower() == 'macaulay':
            return macaulay_duration
        elif duration_type.lower() == 'modified':
            # Modified duration = Macaulay / (1 + ytm)
            # For continuous compounding, Modified ≈ Macaulay
            modified_duration = macaulay_duration
            return modified_duration
        else:
            raise ValueError(f"Unknown duration type: {duration_type}")

###############################################################################

    def convexity(self,
                  settlement_dt: Date,
                  discount_curve: DiscountCurve,
                  z_spread: float = 0.0):
        """
        Calculate bond convexity.

        Convexity measures the curvature of the price-yield relationship.

        Args:
            settlement_dt: Settlement date
            discount_curve: Discount curve
            z_spread: Z-spread in decimal

        Returns:
            Convexity (dimensionless)
        """
        # Get current price and YTM
        clean_px = self.clean_price(settlement_dt, discount_curve, z_spread, settlement_dt)
        ytm = self.yield_to_maturity(settlement_dt, clean_px)

        # Calculate convexity
        weighted_time_squared = 0.0
        total_pv = 0.0

        for i, payment_dt in enumerate(self._payment_dts):
            if payment_dt > settlement_dt:
                time_to_payment = (payment_dt - settlement_dt) / 365.25
                df = np.exp(-ytm * time_to_payment)
                pv = self._coupon_payments[i] * df
                weighted_time_squared += pv * time_to_payment ** 2
                total_pv += pv

        # Principal
        if self._maturity_dt > settlement_dt:
            time_to_maturity = (self._maturity_dt - settlement_dt) / 365.25
            df = np.exp(-ytm * time_to_maturity)
            pv = self._face_value * df
            weighted_time_squared += pv * time_to_maturity ** 2
            total_pv += pv

        convexity = weighted_time_squared / total_pv

        return convexity

###############################################################################

    def dv01(self,
             settlement_dt: Date,
             discount_curve: DiscountCurve,
             z_spread: float = 0.0):
        """
        Calculate DV01 (dollar value of 1 basis point).

        Measures the change in bond value for a 1bp parallel shift in yields.

        Args:
            settlement_dt: Settlement date
            discount_curve: Discount curve
            z_spread: Z-spread in decimal

        Returns:
            DV01 (change in value per 1bp yield change)
        """
        # Bump size: 1bp = 0.0001
        bump = 0.0001

        # Value with down bump
        pv_down = self.value(settlement_dt, discount_curve, z_spread - bump, settlement_dt)

        # Value with up bump
        pv_up = self.value(settlement_dt, discount_curve, z_spread + bump, settlement_dt)

        # DV01 = (PV_down - PV_up) / 2
        dv01 = (pv_down - pv_up) / 2.0

        return dv01

###############################################################################

    def key_rate_durations(self, model):
        """
        Calculate key rate durations from AD delta.

        Key rate duration measures the percentage price sensitivity to a 100bp (1%)
        shift in each specific tenor of the yield curve. This is the standard duration
        convention where a KRD of 4.5 means a 100bp rate increase causes a 4.5% price decline.

        Args:
            model: Model instance with built curves

        Returns:
            dict: Dictionary mapping tenor strings to key rate durations

        Example:
            >>> from cavour.models.models import Model
            >>> model = Model(value_dt)
            >>> model.build_curve("USD_OIS_SOFR", px_list, tenor_list)
            >>> bond = Bond(...)
            >>> krds = bond.key_rate_durations(model)
            >>> print(krds)
            {'1Y': 0.15, '2Y': 0.28, '3Y': 0.42, ...}
        """
        from cavour.market.position.engine import Engine
        from cavour.utils.global_types import RequestTypes

        # Compute delta via engine
        engine = Engine(model)
        result = engine.compute(self, [RequestTypes.VALUE, RequestTypes.DELTA])

        # Get current price
        price = result.value.amount

        # Convert delta to key rate duration
        # Delta is dollar change per 1bp shift (shift = 0.0001 in decimal)
        # Duration D = -(1/P) * (dP/dy) where dy is in decimal (1% = 0.01)
        # Therefore: KRD = -(Delta / P) / 0.0001 = -(Delta / P) * 10000
        krds = {}
        for tenor, delta_val in zip(result.risk.tenors, result.risk.risk_ladder):
            # Duration is positive when rates up -> price down
            # Delta is negative when rates up -> price down
            # Multiply by 10000 to convert from 1bp units to percentage units
            krd = -float(delta_val) / price * 10000.0 if price != 0 else 0.0
            krds[tenor] = krd

        return krds

###############################################################################

    def cs01(self,
             settlement_dt,
             discount_curve,
             z_spread: float = 0.0):
        """
        Calculate CS01 (credit spread 01).

        Measures the change in bond value for a 1bp parallel shift in the
        credit spread (z-spread). This is the standard credit risk measure
        for bonds.

        Args:
            settlement_dt: Settlement date
            discount_curve: Base discount curve
            z_spread: Current z-spread in decimal (default 0)

        Returns:
            CS01 (change in value per 1bp z-spread change)

        Example:
            >>> bond = Bond(...)
            >>> cs01 = bond.cs01(value_dt, curve, z_spread=0.01)  # 100bp current spread
            >>> print(f"CS01: ${cs01:.2f}")
            CS01: $4.25
        """
        # Bump size: 1bp = 0.0001
        bump = 0.0001

        # Value with z-spread down
        pv_down = self.value(settlement_dt, discount_curve, z_spread - bump, settlement_dt)

        # Value with z-spread up
        pv_up = self.value(settlement_dt, discount_curve, z_spread + bump, settlement_dt)

        # CS01 = (PV_down - PV_up) / 2
        # Spread up -> price down, so this is positive
        cs01 = (pv_down - pv_up) / 2.0

        return cs01

###############################################################################

    def print_payments(self):
        """
        Print the bond's payment schedule.

        Shows all coupon payment dates, accrual periods, and amounts.
        """
        print("\n" + "="*80)
        print("BOND PAYMENT SCHEDULE")
        print("="*80)
        print(f"Issue Date:    {self._issue_dt}")
        print(f"Maturity Date: {self._maturity_dt}")
        print(f"Coupon:        {self._coupon*100:.4f}%")
        print(f"Frequency:     {self._freq_type}")
        print(f"Day Count:     {self._dc_type}")
        print(f"Face Value:    {self._face_value:,.2f} {self._currency}")
        print("="*80)

        if self._is_zero_coupon:
            print("ZERO COUPON BOND - No intermediate coupons")
            print(f"Principal at maturity: {self._face_value:,.2f}")
        else:
            print(f"{'Num':<5} {'Accrual Start':<15} {'Accrual End':<15} "
                  f"{'Payment Date':<15} {'Year Frac':<12} {'Coupon':<15}")
            print("-"*80)

            for i in range(self._num_coupons):
                print(f"{i+1:<5} "
                      f"{str(self._accrual_start_dts[i]):<15} "
                      f"{str(self._accrual_end_dts[i]):<15} "
                      f"{str(self._payment_dts[i]):<15} "
                      f"{self._year_fracs[i]:<12.6f} "
                      f"{self._coupon_payments[i]:>15,.2f}")

        print("-"*80)
        print(f"Principal payment at maturity: {self._face_value:,.2f}")
        print("="*80)

###############################################################################

    def print_valuation(self,
                        value_dt: Date,
                        discount_curve: DiscountCurve,
                        z_spread: float = 0.0,
                        settlement_dt: Date = None):
        """
        Print detailed valuation showing PV of each cashflow.

        Args:
            value_dt: Valuation date
            discount_curve: Discount curve
            z_spread: Z-spread in decimal
            settlement_dt: Settlement date
        """
        if settlement_dt is None:
            settlement_dt = value_dt

        print("\n" + "="*90)
        print("BOND VALUATION")
        print("="*90)
        print(f"Value Date:     {value_dt}")
        print(f"Settlement Date: {settlement_dt}")
        print(f"Z-Spread:       {z_spread*10000:.2f} bp")
        print("="*90)

        df_settlement = discount_curve.df(settlement_dt)
        total_pv = 0.0

        print(f"{'Num':<5} {'Payment Date':<15} {'Coupon':<15} "
              f"{'DF':<12} {'PV':<15}")
        print("-"*90)

        # Value coupons
        for i, payment_dt in enumerate(self._payment_dts):
            if payment_dt > settlement_dt:
                df_payment = discount_curve.df(payment_dt)

                if z_spread != 0.0:
                    time_to_payment = (payment_dt - settlement_dt) / 365.25
                    df_payment = df_payment * np.exp(-z_spread * time_to_payment)

                df_rel = df_payment / df_settlement
                pv = self._coupon_payments[i] * df_rel
                total_pv += pv

                print(f"{i+1:<5} "
                      f"{str(payment_dt):<15} "
                      f"{self._coupon_payments[i]:>15,.2f} "
                      f"{df_rel:<12.8f} "
                      f"{pv:>15,.2f}")

        # Value principal
        if self._maturity_dt > settlement_dt:
            df_maturity = discount_curve.df(self._maturity_dt)

            if z_spread != 0.0:
                time_to_maturity = (self._maturity_dt - settlement_dt) / 365.25
                df_maturity = df_maturity * np.exp(-z_spread * time_to_maturity)

            df_rel = df_maturity / df_settlement
            principal_pv = self._face_value * df_rel
            total_pv += principal_pv

            print(f"{'PRIN':<5} "
                  f"{str(self._maturity_dt):<15} "
                  f"{self._face_value:>15,.2f} "
                  f"{df_rel:<12.8f} "
                  f"{principal_pv:>15,.2f}")

        print("-"*90)
        print(f"{'TOTAL PV:':<37} {total_pv:>15,.2f}")

        # Calculate prices
        dirty_px = (total_pv / self._face_value) * 100.0
        accrued = self.accrued_interest(settlement_dt)
        accrued_per_100 = (accrued / self._face_value) * 100.0
        clean_px = dirty_px - accrued_per_100

        print(f"{'Dirty Price:':<37} {dirty_px:>15.4f}")
        print(f"{'Accrued Interest:':<37} {accrued_per_100:>15.4f}")
        print(f"{'Clean Price:':<37} {clean_px:>15.4f}")
        print("="*90)

###############################################################################

    def __repr__(self):
        """String representation of the bond."""
        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("ISSUE DATE", self._issue_dt)
        s += label_to_string("MATURITY DATE", self._maturity_dt)
        s += label_to_string("COUPON", f"{self._coupon*100:.4f}%")
        s += label_to_string("FREQUENCY", self._freq_type)
        s += label_to_string("DAY COUNT", self._dc_type)
        s += label_to_string("CURRENCY", self._currency)
        s += label_to_string("FACE VALUE", self._face_value)

        if self._is_zero_coupon:
            s += label_to_string("TYPE", "ZERO COUPON BOND")
        else:
            s += label_to_string("NUMBER OF COUPONS", self._num_coupons)

        return s

###############################################################################

    def _print(self):
        """Print the bond details."""
        print(self)


###############################################################################

    @staticmethod
    def generate_equal_principal_schedule(face_value: float, num_periods: int):
        """
        Generate an equal principal amortization schedule.

        Principal is repaid in equal amounts at each period.

        Args:
            face_value: Initial principal amount
            num_periods: Number of payment periods

        Returns:
            List of outstanding principal at each payment date
            (excludes initial face value)

        Example:
            >>> Bond.generate_equal_principal_schedule(100, 4)
            [75.0, 50.0, 25.0, 0.0]
        """
        if num_periods <= 0:
            raise LibError("Number of periods must be positive")

        principal_payment = face_value / num_periods
        schedule = []

        for i in range(1, num_periods + 1):
            remaining = face_value - (i * principal_payment)
            schedule.append(max(0.0, remaining))  # Ensure no negative values

        return schedule

    @staticmethod
    def generate_annuity_schedule(face_value: float, num_periods: int, coupon_rate: float, freq_type: FrequencyTypes):
        """
        Generate an annuity (constant payment) amortization schedule.

        Total payment (coupon + principal) is constant each period.
        Principal repayment increases over time as interest decreases.

        Args:
            face_value: Initial principal amount
            num_periods: Number of payment periods
            coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
            freq_type: Payment frequency (for periodic rate calculation)

        Returns:
            List of outstanding principal at each payment date

        Note:
            Uses the annuity formula to calculate constant payment amount,
            then derives principal repayment schedule.
        """
        if num_periods <= 0:
            raise LibError("Number of periods must be positive")

        # Convert annual rate to periodic rate based on frequency
        from cavour.utils.frequency import FrequencyTypes
        freq_map = {
            FrequencyTypes.ANNUAL: 1,
            FrequencyTypes.SEMI_ANNUAL: 2,
            FrequencyTypes.QUARTERLY: 4,
            FrequencyTypes.MONTHLY: 12
        }

        periods_per_year = freq_map.get(freq_type, 1)
        periodic_rate = coupon_rate / periods_per_year

        if periodic_rate == 0:
            # If rate is zero, equal principal amortization
            return Bond.generate_equal_principal_schedule(face_value, num_periods)

        # Calculate constant payment using annuity formula
        # PMT = P * [r(1+r)^n] / [(1+r)^n - 1]
        import math
        factor = (1 + periodic_rate) ** num_periods
        constant_payment = face_value * (periodic_rate * factor) / (factor - 1)

        # Build schedule by tracking remaining balance
        schedule = []
        balance = face_value

        for _ in range(num_periods):
            interest_payment = balance * periodic_rate
            principal_payment = constant_payment - interest_payment
            balance -= principal_payment
            schedule.append(max(0.0, balance))  # Ensure no negative values

        return schedule

###############################################################################

###############################################################################
