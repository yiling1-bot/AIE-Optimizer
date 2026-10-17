"""Structured failures for analyses that cannot produce a trustworthy timing."""


class AnalysisError(ValueError):
    category = "invalid_input"

    def __init__(self, code, message, context=None):
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})

    def as_dict(self):
        return {
            "category": self.category,
            "code": self.code,
            "message": str(self),
            "context": self.context,
        }


class UnsupportedAnalysis(AnalysisError):
    category = "unsupported"


class InvalidAnalysisInput(AnalysisError):
    category = "invalid_input"
