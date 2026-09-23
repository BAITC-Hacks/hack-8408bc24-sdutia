"""User-facing error types. Each carries an exit code for the CLI and RU/EN messages for the UI."""


class WindAgentError(Exception):
    exit_code = 1

    def __init__(self, message_en: str, message_ru: str | None = None):
        super().__init__(message_en)
        self.user_message_en = message_en
        self.user_message_ru = message_ru or message_en


class InputError(WindAgentError):
    """Invalid user input: bad date, unknown site, wrong file format."""

    exit_code = 2


class DataUnavailableError(WindAgentError):
    """Required data or generated outputs are missing."""

    exit_code = 3


class ExternalServiceError(WindAgentError):
    """An external API failed and no cached data could replace it."""

    exit_code = 4
