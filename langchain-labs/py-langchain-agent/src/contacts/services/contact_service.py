from uuid import UUID

from src.contacts.database import get_db, SessionFactory
from src.contacts.domain.contact import Contact

from src.contacts.repositories.contact_repository import ContactRepository
from src.contacts.repositories.sqlalchemy_contact_repository import SQLAlchemyContactRepository


class ContactService:
    def __init__(self):
        self.session = SessionFactory()
        self.repo: ContactRepository = SQLAlchemyContactRepository(self.session)

    def get_contact(self, id: UUID) -> Contact | None:
        return self.repo.get_by_id(contact_id=id)