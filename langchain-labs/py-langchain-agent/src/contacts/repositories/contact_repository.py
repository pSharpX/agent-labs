
from abc import ABC, abstractmethod
from uuid import UUID

from src.contacts.domain.contact import Contact


class ContactRepository(ABC):

    @abstractmethod
    def get_by_id(self, contact_id: UUID) -> Contact | None:
        pass

    @abstractmethod
    def find_by_name(self, name: str) -> list[Contact]:
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> Contact | None:
        pass

    @abstractmethod
    def find_by_phone(self, phone_number: str) -> Contact | None:
        pass

    @abstractmethod
    def find_by_nationality(
        self,
        nationality: str,
    ) -> list[Contact]:
        pass

    @abstractmethod
    def create(self, contact: Contact) -> Contact:
        pass

    @abstractmethod
    def update(self, contact: Contact) -> Contact:
        pass

    @abstractmethod
    def delete(self, contact_id: UUID) -> None:
        pass