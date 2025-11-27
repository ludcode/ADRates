# XCCY Curve Construction from Two OIS Curves and a Basis Curve

## 0. Goal

Build a cross-currency (XCCY) discount curve for the foreign currency, **discounted in the domestic collateral currency**, such that:

- Inputs:
  - Domestic OIS discount curve
  - Foreign OIS discount curve
  - Cross-currency basis curve (tenor -> spread)
- Output:
  - A "foreign-in-domestic" XCCY discount curve P_f_d(T)
- Condition:
  - For each basis tenor T_k, a standard XCCY float-float swap built
    using the same basis quote, schedules, calendars, and conventions
    must have PV = 0 in domestic currency (within numerical tolerance).


## 1. Notation

- t0                = valuation date
- S0                = spot FX rate (domestic per 1 unit of foreign)
- d                 = domestic currency
- f                 = foreign currency

Curves:

- P_d(T)            = domestic OIS discount factor at time T
                      (domestic cashflows discounted in domestic collateral)

- P_f_f(T)          = foreign OIS discount factor at time T
                      (foreign cashflows discounted in foreign collateral)

- P_f_d(T)          = foreign-in-domestic XCCY discount factor at time T
                      (foreign cashflows discounted under domestic collateral)
                      -> THIS is the curve we are bootstrapping.

Basis curve:

- Basis pillars: (T_k, B_k_bp), for k = 1..N
  - T_k             = maturity (tenor converted to actual date)
  - B_k_bp          = quoted basis spread in basis points per annum
  - B_k             = B_k_bp / 10_000 (decimal)


## 2. Required Inputs

### 2.1 Curves

The code should accept curve objects with at least a discount-factor API:

- Domestic OIS curve:
  - df_d(T: date) -> float
  - returns P_d(T)

- Foreign OIS curve:
  - df_f_foreign_collateral(T: date) -> float
  - returns P_f_f(T)

The XCCY curve we will construct:

- df_f_in_domestic(T: date) -> float
  - returns P_f_d(T)


### 2.2 FX and Collateral

- spot_fx = S0 (domestic per 1 unit foreign)
- collateral_currency = "domestic"
  - assume all basis swaps are collateralised in domestic currency


### 2.3 Calendars and Conventions

All conventions MUST be explicitly configurable and consistent with
how the market basis swaps and OIS curves are defined.

Per-currency conventions:

- Domestic currency (d):
  - calendar_d                  (e.g. "London")
  - ois_dcc_d                   (e.g. "ACT/365F")
  - ois_frequency_d             (e.g. "1Y" or "3M")
  - bus_day_conv_d              (e.g. "MODIFIED_FOLLOWING")
  - spot_lag_d                  (in business days, e.g. 2)

- Foreign currency (f):
  - calendar_f                  (e.g. "TARGET" or "NewYork")
  - ois_dcc_f
  - ois_frequency_f
  - bus_day_conv_f
  - spot_lag_f

Pair-level conventions:

- pair_calendar                 (joint or union calendar of d and f)
- fx_spot_lag                   (business days on pair_calendar, usually 2)
- roll_convention               (e.g. "NONE" or "EOM")
- stub_type                     (e.g. "SHORT_FRONT", "LONG_BACK")

Leg-level conventions (for the basis swap itself):

- For domestic leg:
  - dcc_leg_d                   (day-count for floating leg)
  - freq_leg_d                  (coupon frequency, e.g. "3M", "6M")
  - pay_lag_d                   (payment lag in days)
  - index_type_d                (e.g. OIS, IBOR, etc.)

- For foreign leg:
  - dcc_leg_f
  - freq_leg_f
  - pay_lag_f
  - index_type_f

The **same** schedule and convention logic must be used:

- when bootstrapping, and
- when later pricing a basis swap to check PV = 0.

Any mismatch in calendars, day-counts, business-day conventions,
spot lags, stub rules, or which leg carries the basis will break
the PV = 0 identity.


### 2.4 Basis Quote Side and Sign

The implementation must explicitly encode:

- basis_leg:
  - "domestic" or "foreign"
  - which leg has the extra spread added to its index.

- basis_sign_convention:
  - for example, market convention might be:
    "receive domestic, pay foreign + spread"
  - internally, define a consistent sign:

    Example:
    - Domestic leg coupon rate: F_d
    - Foreign leg coupon rate: F_f + B_k

  where B_k is positive if the foreign leg pays more than its pure
  index, according to the market quote.

This must be parameterised, not hard-coded, so that the same logic
can be reused for different currency pairs and platforms.


## 3. Calibration Instrument: XCCY Basis Swap

For each basis pillar (T_k, B_k), define a standard XCCY float-float swap.

### 3.1 High-level structure

- Start date: XCCY spot date T_spot
  - T_spot = t0 + fx_spot_lag business days on pair_calendar

- Maturity date: T_k
  - obtained by adding tenor (e.g. "1Y", "2Y") to T_spot
    and rolling with pair_calendar, bus_day_conv, and roll_convention.

- Legs:
  - Domestic leg in currency d
  - Foreign leg in currency f with basis B_k (or vice versa, depending on config)

- Notional:
  - Foreign notional: N_f = 1.0 (one unit of foreign)
  - Domestic notional: N_d = S0 * N_f

  This choice makes the initial notional exchange worth approximately zero
  in domestic currency at T_spot.


### 3.2 Schedules

For each leg L in {d, f}, generate a schedule:

- Use:
  - start date: T_spot
  - end date: T_k
  - frequency: freq_leg_L
  - calendar: calendar_L
  - business-day convention: bus_day_conv_L
  - roll_convention and stub_type

- Produce:
  - accrual start dates: t_start_L[i]
  - accrual end dates:   t_end_L[i]
  - payment dates:       pay_L[i]
  - accrual factors:     alpha_L[i] = year_fraction(t_start_L[i], t_end_L[i], dcc_leg_L)


### 3.3 Floating rates

Floating rates for each leg are derived from the corresponding OIS curve
in its own collateral currency:

- Domestic leg forward rate for period i:

  F_d[i] = ( 1 / alpha_d[i] ) * ( P_d( t_start_d[i] ) / P_d( t_end_d[i] ) - 1 )

- Foreign leg forward rate for period j:

  F_f[j] = ( 1 / alpha_f[j] ) * ( P_f_f( t_start_f[j] ) / P_f_f( t_end_f[j] ) - 1 )

Note: P_d and P_f_f are the input OIS curves.
Do NOT use the yet-to-be-constructed XCCY curve for forwards.
The XCCY curve is only used for discounting foreign cashflows
in domestic collateral.


### 3.4 Coupons

Example: basis spread is applied to the foreign leg.

- Domestic leg coupon in period i:

  CF_d[i] = sign_d * N_d * F_d[i] * alpha_d[i]

- Foreign leg coupon in period j:

  CF_f[j] = sign_f * N_f * ( F_f[j] + B_k ) * alpha_f[j]

where:

- sign_d, sign_f encode whether the swap is payer or receiver from the
  domestic PV perspective.
- B_k is the decimal spread: B_k = B_k_bp / 10_000.

The implementation must keep a consistent sign convention and document it.


### 3.5 Notional exchanges

At T_spot:

- One leg pays N_f in foreign, receives N_d in domestic.
- The other leg receives N_f in foreign, pays N_d in domestic.

At T_k (maturity):

- Notionals are exchanged back:

  - Leg that received N_f at T_spot now pays N_f at T_k.
  - Leg that received N_d at T_spot now pays N_d at T_k.

Implementation detail: treat notionals as additional cashflows:

- Domestic notional CFs in domestic currency
- Foreign notional CFs in foreign currency

Foreign notionals will be discounted via P_f_d(T) and converted at S0.


## 4. PV Computation Using XCCY Curve

All PVs are expressed in domestic currency.

### 4.1 Domestic leg PV

Domestic leg PV is fully determined by the domestic OIS curve and
domestic notional CFs:

PV_d = sum_i [ CF_d[i] * P_d( pay_d[i] ) ] + PV_domestic_notional_exchanges

No XCCY curve is involved here.


### 4.2 Foreign leg PV using XCCY discounting

Foreign leg cashflows are:

- Determined using the foreign OIS curve (for F_f[i])
- Discounted using the foreign-in-domestic curve P_f_d(T)
- Converted to domestic using S0

Define:

PV_f_dom = S0 * PV_f_d

where:

PV_f_d = sum_j [ CF_f[j] * P_f_d( pay_f[j] ) ] + PV_foreign_notional_exchanges_d

Foreign notional exchanges (at T_spot and T_k) are also discounted
with P_f_d and included in PV_f_d.


### 4.3 Par condition for calibration

For the k-th XCCY swap (pillar T_k, spread B_k), the par condition is:

PV_d(k) + PV_f_dom(k) = 0

or explicitly:

PV_d(k) + S0 * PV_f_d(k) = 0

This equation will determine one unknown new discount factor:

P_f_d( T_k )


## 5. Bootstrapping Algorithm

We bootstrap P_f_d(T) by increasing maturity of the basis pillars:

T_1 < T_2 < ... < T_N

Data structure for the curve under construction:

- xccy_nodes: map from date T -> P_f_d(T)

Initialize:

- xccy_nodes[t0] = 1.0


### 5.1 Key idea

Consider the XCCY basis swap with maturity T_k.

- Many foreign cashflows (coupons, possibly notional exchanges if spot or stubs)
  occur at dates < T_k.
- Discount factors P_f_d for those earlier dates have already been bootstrapped
  from previous pillars.
- Only the cashflows at T_k depend on the new unknown discount factor P_f_d(T_k).

Therefore the PV equation for this swap is linear in P_f_d(T_k) and has a
closed-form solution, no numeric root finder is needed.


### 5.2 Decomposition of the foreign leg PV

For pillar k:

1. Build the full XCCY basis swap (domestic and foreign legs) with
   maturity T_k using the conventions and basis spread B_k.

2. Compute the domestic leg PV:

   PV_d = sum_i [ CF_d[i] * P_d( pay_d[i] ) ] + PV_domestic_notional_exchanges

3. Split foreign CFs into two sets:

   - J_before: indices j where pay_f[j] < T_k
   - J_at:     indices j where pay_f[j] = T_k

   Usually J_at includes:
   - last coupon, and
   - foreign notional exchange.

4. Known part of foreign PV (dates < T_k):

   PV_f_known_d = sum_{j in J_before} [ CF_f[j] * P_f_d( pay_f[j] ) ]
                  + PV_foreign_notional_exchanges_before_Tk

   - All P_f_d(pay_f[j]) in this sum are known from previous steps.
   - If a pay date < T_k is not an existing node, interpolation using
     the chosen scheme should be applied consistently.

5. Unknown part of foreign PV at T_k:

   Aggregate all foreign cashflows at T_k into one number:

   CF_last_f = sum_{j in J_at} CF_f[j] + (foreign notional at T_k, if not
                already included in CF_f[j])

   Then:

   PV_f_unknown_d = CF_last_f * P_f_d( T_k )

6. Total foreign PV in foreign-discounted-in-domestic terms:

   PV_f_d_total = PV_f_known_d + CF_last_f * P_f_d( T_k )


### 5.3 Solving for P_f_d(T_k)

The par condition is:

PV_d + S0 * PV_f_d_total = 0

Substitute PV_f_d_total:

PV_d + S0 * ( PV_f_known_d + CF_last_f * P_f_d(T_k) ) = 0

Solve for P_f_d(T_k):

P_f_d(T_k) = - ( PV_d + S0 * PV_f_known_d ) / ( S0 * CF_last_f )

This gives the new XCCY discount factor for maturity T_k.

Store:

xccy_nodes[T_k] = P_f_d(T_k)


### 5.4 Iteration over all pillars

Algorithm sketch:

1. Sort basis pillars by maturity date T_k ascending.
2. For each pillar (T_k, B_k):

   - Build XCCY basis swap schedules and cashflows with spread B_k.
   - Compute PV_d.
   - Compute PV_f_known_d using existing xccy_nodes and interpolation.
   - Compute CF_last_f at T_k.
   - Compute P_f_d(T_k) from the formula above.
   - Add xccy_nodes[T_k] = P_f_d(T_k).

3. After all pillars are processed, construct a curve object
   df_f_in_domestic(T) from xccy_nodes using the chosen interpolation.


## 6. Curve Representation and Interpolation

To preserve calibration, the curve representation must:

- Preserve node values exactly at T_k.
- Provide a stable interpolation for other dates between pillars.

Recommended options:

- log-linear on discount factors:
  - work with y(T) = log(P_f_d(T)), interpolate y(T) linearly in time,
    then P_f_d(T) = exp(y(T)).

- or linear on continuously compounded zero rates:
  - define z(T) = -log(P_f_d(T)) / T, interpolate z(T) linearly in T.

Configuration parameter:

- xccy_interp_mode = "log_df" or "linear_zero"

The same interpolation mode must be used consistently:

- during bootstrapping (whenever discounting CFs at dates that are not
  direct nodes); and
- during pricing of instruments with the finished curve.


## 7. Pricing New XCCY Swaps With the Bootstrapped Curve

To price any XCCY swap after building df_f_in_domestic:

1. Build domestic and foreign schedules using the same conventions as for
   calibration.

2. Domestic leg:
   - Compute forwards F_d using P_d (domestic OIS).
   - Compute coupons CF_d[i].
   - Discount all domestic CFs with P_d(T).

3. Foreign leg:
   - Compute forwards F_f using P_f_f (foreign OIS).
   - Build coupons CF_f[j] including any basis.
   - Discount foreign CFs with P_f_d(T) = df_f_in_domestic(T).
   - Convert to domestic currency by S0.

4. PV = PV_d + S0 * PV_f_d

For the **calibration instruments** (same T_k and basis B_k used in the bootstrap), 
PV must be approximately 0 in domestic currency (within tolerance).


## 8. Sanity Checks and Tests

After building the XCCY curve, run at least the following tests:

1. **Reprice calibration swaps**

   For each pillar (T_k, B_k):

   - Rebuild the XCCY basis swap using:
     - same T_spot, T_k
     - same calendars and business-day conventions
     - same day-counts and frequencies
     - same stub rules and pay_lags
     - same leg that carries the basis
     - same basis spread B_k

   - Price it with:
     - domestic OIS curve P_d
     - foreign OIS curve P_f_f
     - XCCY curve P_f_d
     - spot FX S0

   - Check:

     abs( PV_domestic ) < tolerance

     Example tolerance:
       - 1e-10 relative to notional, or
       - a few micro currency units.


2. **Small perturbation check**

   - For a given pillar T_k, price the same swap with basis B_k + 1bp.
   - The PV should be approximately equal to the (domestic) DV01 of the
     leg that carries the spread (up to interpolation effects).

3. **Consistency of conventions**

   If PV != 0 for calibration swaps, first check:

   - Are the schedules for calibration and pricing literally identical?
   - Are calendars, day-counts, business-day conventions, spot lags,
     roll conventions, stub types identical?
   - Is the basis applied to the same leg with the same sign?
   - Are notionals set consistently (N_f = 1, N_d = S0)?
   - Are foreign cashflows discounted with P_f_d, not with P_f_f?

   Any difference here will show up as a non-zero PV at pillars.
