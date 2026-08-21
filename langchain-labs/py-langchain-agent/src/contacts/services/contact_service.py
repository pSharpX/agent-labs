from uuid import UUID

from src.contacts.database import SessionFactory
from src.contacts.domain.contact import Contact

from src.contacts.repositories.contact_repository import ContactRepository
from src.contacts.repositories.sqlalchemy_contact_repository import SQLAlchemyContactRepository


class ContactService:
    def __init__(self):
        self.session = SessionFactory()
        self.repo: ContactRepository = SQLAlchemyContactRepository(self.session)

    def get_contact(self, contact_id: str) -> Contact | None:
        return self.repo.get_by_id(contact_id=UUID(contact_id))

    def search_contact(self, name: str) -> list[Contact]:
        return self.repo.find_by_name(name=name)

    def update_contact(self, contact: Contact) -> Contact:
        return self.repo.update(contact=contact)