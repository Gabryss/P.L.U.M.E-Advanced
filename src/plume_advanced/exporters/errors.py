"""Export failures that require an export change, not another procedural seed."""

class ExportBudgetError(ValueError):
    """An explicit triangle, surface-error or package-size budget was exhausted."""

    def __init__(self, message, *, report=None):
        self.report = report or dict(passed=False, failures=[message])
        super().__init__("; ".join(self.report["failures"]))
