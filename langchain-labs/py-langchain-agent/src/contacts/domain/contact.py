
from datetime import date
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Contact(BaseModel):
    """Represents a person's contact information and personal preferences."""
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        description="Unique identifier of the contact."
    )
    first_name: str = Field(
        description="The contact's first/given name."
    )
    last_name: str | None = Field(
        default=None,
        description="The contact's last/family name."
    )
    display_name: str | None = Field(
        default=None,
        description=(
            "The preferred name to use when addressing or referring to "
            "the contact. For example, 'Chris' instead of 'Christian Rivera'."
        ),
    )

    birthday: date | None = Field(
        default=None,
        description="The contact's date of birth.",
    )
    nationality: str | None = Field(
        default=None,
        description="The contact's nationality.",
    )

    email: str | None = Field(
        default=None,
        description="The contact's primary email address.",
    )
    phone_number: str | None = Field(
        default=None,
        description="The contact's primary phone number.",
    )

    address_line_1: str | None = Field(
        default=None,
        description="The primary street address of the contact.",
    )
    address_line_2: str | None = Field(
        default=None,
        description="Additional address information such as apartment, "
                    "unit, floor, or building number.",
    )
    city: str | None = Field(
        default=None,
        description="The city where the contact lives or is located.",
    )
    country: str | None = Field(
        default=None,
        description="The country where the contact lives or is located.",
    )

    company: str | None = Field(
        default=None,
        description="The company or organization where the contact works.",
    )
    job_title: str | None = Field(
        default=None,
        description="The contact's professional job title or role.",
    )

    social_networks: dict[str, str] | None = Field(
        default=None,
        description=(
            "Social network accounts belonging to the contact. "
            "Use the social network name as the key and the profile "
            "identifier or URL as the value. For example: "
            "{'linkedin': 'https://linkedin.com/in/john-doe'}."
        ),
    )
    preferences: dict[str, Any] | None = Field(
        default=None,
        description=(
            "User-specific preferences associated with the contact. "
            "Store structured preference information as key-value pairs."
        ),
    )

    notes: str | None = Field(
        default=None,
        description=(
            "Free-form notes or additional relevant information "
            "about the contact."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional structured information that does not belong "
            "to any of the standard contact fields."
        ),
    )