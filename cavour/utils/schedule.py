
from .error import LibError
from .date import Date
from .calendar import (Calendar, CalendarTypes)
from .calendar import (BusDayAdjustTypes, DateGenRuleTypes)
from .frequency import (annual_frequency, FrequencyTypes)
from .helpers import label_to_string
from .helpers import check_argument_types


class Schedule:
    """ A schedule is a set of dates generated according to ISDA standard
    rules which starts on the next date after the effective date and runs up to
    a termination date. Dates are adjusted to a provided calendar. The zeroth
    element is the previous coupon date (PCD) and the first element is the
    Next Coupon Date (NCD). We reference ISDA 2006."""

    def __init__(self,
                 effective_dt: Date,  # Also known as the start date
                 # This is UNADJUSTED (set flag to adjust it)
                 termination_dt: Date,
                 freq_type: FrequencyTypes = FrequencyTypes.ANNUAL,
                 cal_type: CalendarTypes = CalendarTypes.WEEKEND,
                 bd_type: BusDayAdjustTypes = BusDayAdjustTypes.FOLLOWING,
                 dg_type: DateGenRuleTypes = DateGenRuleTypes.BACKWARD,
                 adjust_termination_dt: bool = True,  # Default is to adjust
                 end_of_month: bool = False,  # All flow dates are EOM if True
                 first_dt=None,  # First coupon date
                 next_to_last_dt=None):  # Penultimate coupon date
        """ Create Schedule object which calculates a sequence of dates
        following the ISDA convention for fixed income products, mainly swaps.

        If the date gen rule type is FORWARD we get the unadjusted dates by
        stepping forward from the effective date in steps of months determined
        by the period tenor - i.e. the number of months between payments. We
        stop before we go past the termination date.

        If the date gen rule type is BACKWARD we get the unadjusted dates by
        stepping backward from the termination date in steps of months
        determined by the period tenor - i.e. the number of months between
        payments. We stop before we go past the effective date.

        - If the EOM flag is false, and the start date is on the 31st then the
        the unadjusted dates will fall on the 30 if a 30 is a previous date.
        - If the EOM flag is false and the start date is 28 Feb then all
        unadjusted dates will fall on the 28th.
        - If the EOM flag is false and the start date is 28 Feb then all
        unadjusted dates will fall on their respective EOM.

        We then adjust all of the flow dates if they fall on a weekend or
        holiday according to the calendar specified. These dates are adjusted
        in accordance with the business date adjustment.

        The effective date is never adjusted as it is not a payment date.
        The termination date is not automatically business day adjusted in a
        swap - assuming it is a holiday date. This must be explicitly stated in
        the trade confirm. However, it is adjusted in a CDS contract as
        standard.

        Inputs first_dt and next_to_last_dt are for managing long payment
        stubs at the start and end of the swap but *have not yet been
        implemented*. All stubs are currently short, either at the start or
        end of swap. """

        check_argument_types(self.__init__, locals())

        if effective_dt >= termination_dt:
            raise LibError("Effective date must be before termination date.")

        self._effective_dt = effective_dt
        self._termination_dt = termination_dt

        if first_dt is None:
            self._first_dt = effective_dt
        else:
            if first_dt > effective_dt and first_dt < termination_dt:
                self._first_dt = first_dt
                print("FIRST DATE NOT IMPLEMENTED")  # TODO
            else:
                raise LibError("First date must be after effective date and" +
                               " before termination date")

        if next_to_last_dt is None:
            self._next_to_last_dt = termination_dt
        else:
            if next_to_last_dt > effective_dt and next_to_last_dt < termination_dt:
                self._next_to_last_dt = next_to_last_dt
                print("NEXT TO LAST DATE NOT IMPLEMENTED")  # TODO
            else:
                raise LibError("Next to last date must be after effective date and" +
                               " before termination date")

        self._freq_type = freq_type
        self._cal_type = cal_type
        self._bd_type = bd_type
        self._dg_type = dg_type

        self._adjust_termination_dt = adjust_termination_dt

        if end_of_month is True:
            self._end_of_month = True
        else:
            self._end_of_month = False

        self._adjusted_dts = None

        self.generate()

    ###########################################################################

    def schedule_dts(self):
        """ Returns a list of the schedule of Dates. """

        if self._adjusted_dts is None:
            self.generate()

        return self._adjusted_dts

    ###########################################################################

    def generate(self):
        """ Generate schedule of dates according to specified date generation
        rules and also adjust these dates for holidays according to the
        specified business day convention and the specified calendar. """

        calendar = Calendar(self._cal_type)
        frequency = annual_frequency(self._freq_type)
        num_months = int(12 / frequency)

        unadjusted_schedule_dts = []
        self._adjusted_dts = []

        if self._dg_type == DateGenRuleTypes.BACKWARD:

            next_dt = self._termination_dt
            flow_num = 0

            while next_dt > self._effective_dt:

                unadjusted_schedule_dts.append(next_dt)
                tot_num_months = num_months * (1 + flow_num)
                next_dt = self._termination_dt.add_months(-tot_num_months)

                if self._end_of_month is True:
                    next_dt = next_dt.eom()

                flow_num += 1

            # Add on the Previous Coupon Date
            unadjusted_schedule_dts.append(next_dt)
            flow_num += 1

            # reverse order and holiday adjust dates
            # the first date is not adjusted as this was provided
            dt = unadjusted_schedule_dts[flow_num - 1]
            self._adjusted_dts.append(dt)

            # We adjust all flows after the effective date and before the
            # termination date to fall on business days according to their cal
            for i in range(1, flow_num - 1):
                dt = calendar.adjust(unadjusted_schedule_dts[flow_num - i - 1],
                                     self._bd_type)

                self._adjusted_dts.append(dt)

            self._adjusted_dts.append(self._termination_dt)

        elif self._dg_type == DateGenRuleTypes.FORWARD:

            # This needs checking
            next_dt = self._effective_dt
            flow_num = 0

            unadjusted_schedule_dts.append(next_dt)
            flow_num = 1

            while next_dt < self._termination_dt:
                unadjusted_schedule_dts.append(next_dt)
                tot_num_months = num_months * (flow_num)
                next_dt = self._effective_dt.add_months(tot_num_months)
                flow_num = flow_num + 1

            # The effective date is not adjusted as it is given
            for i in range(1, flow_num):
                dt = calendar.adjust(unadjusted_schedule_dts[i],
                                     self._bd_type)

                self._adjusted_dts.append(dt)

            self._adjusted_dts.append(self._termination_dt)

        if self._adjusted_dts[0] < self._effective_dt:
            self._adjusted_dts[0] = self._effective_dt

        # The market standard for swaps is not to adjust the termination date
        # unless it is specified in the contract. It is standard for CDS.
        # We change it if the adjust_termination_dt flag is True.
        if self._adjust_termination_dt is True:

            self._termination_dt = calendar.adjust(self._termination_dt,
                                                   self._bd_type)

            self._adjusted_dts[-1] = self._termination_dt

        #######################################################################
        # Check the resulting schedule to ensure that no two dates are the
        # same in which case we remove the duplicate and that they are
        # monotonic - this should never happen but ...
        #######################################################################

        if len(self._adjusted_dts) < 2:
            raise LibError("Schedule has two dates only.")

        prev_dt = self._adjusted_dts[0]
        for dt in self._adjusted_dts[1:]:

            # if the first date lands on the effective date then remove it
            if dt == prev_dt:
                self._adjusted_dts.pop(0)

            if dt < prev_dt:  # Dates must be ordered
                raise LibError("Dates are not monotonic")

            prev_dt = dt

        #######################################################################

        return self._adjusted_dts

    ###########################################################################

    def __repr__(self):
        """ Print out the details of the schedule and the actual dates. This
        can be used for providing transparency on schedule calculations. """

        s = label_to_string("OBJECT TYPE", type(self).__name__)
        s += label_to_string("EFFECTIVE DATE", self._effective_dt)
        s += label_to_string("END DATE", self._termination_dt)
        s += label_to_string("FREQUENCY", self._freq_type)
        s += label_to_string("CALENDAR", self._cal_type)
        s += label_to_string("BUSDAYRULE", self._bd_type)
        s += label_to_string("DATEGENRULE", self._dg_type)
        s += label_to_string("ADJUST TERM DATE", self._adjust_termination_dt)
        s += label_to_string("END OF MONTH", self._end_of_month, "")

        if 1 == 0:
            if len(self._adjusted_dts) > 0:
                s += "\n\n"
                s += label_to_string("EFF", self._adjusted_dts[0], "")

            if len(self._adjusted_dts) > 1:
                s += "\n"
                s += label_to_string("FLW", self._adjusted_dts[1:], "",
                                     list_format=True)

        return s

    ###########################################################################

    def _print(self):
        """ Print out the details of the schedule and the actual dates. This
        can be used for providing transparency on schedule calculations. """
        print(self)

###############################################################################