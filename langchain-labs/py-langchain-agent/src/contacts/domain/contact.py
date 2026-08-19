
from dataclasses import dataclass, field
from datetime import date
from uuid import UUID


@dataclass
class Contact:
    id: UUID
    first_name: str
    last_name: str | None = None
    display_name: str | None = None

    birthday: date | None = None
    nationality: str | None = None

    email: str | None = None
    phone_number: str | None = None

    address_line_1: str | None = None
    address_line_2: str | None = None
    city: str | None = None
    country: str | None = None

    company: str | None = None
    job_title: str | None = None

    social_networks: dict | None = field(default=None)
    preferences: dict | None = field(default=None)

    notes: str | None = None
    metadata: dict | None = field(default=None)