from cavour.trades.rates.ois_curve import OISCurve
from cavour.trades.rates.ois import OIS
try:
    from cavour.trades.rates.xccy_curve import XCCYCurve
except ImportError:
    # xccy_curve may not exist in main library, only in experimental
    pass