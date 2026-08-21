from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from src.contacts.domain.contact import Contact
from src.contacts.infrastructure.contact_mapper import ContactMapper
from src.contacts.repositories.contact_repository import ContactRepository
from src.contacts.models.contact_model import ContactModel


class SQLAlchemyContactRepository(ContactRepository):

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(
        self,
        contact_id: UUID,
    ) -> Contact | None:

        statement = (
            select(ContactModel)
            .where(ContactModel.id == contact_id)
        )

        model = self.session.scalar(statement)
        return ContactMapper.to_domain(model) if model else None

    def find_by_name(
        self,
        name: str,
    ) -> list[Contact]:

        search = f"%{name}%"

        statement = (
            select(ContactModel)
            .where(
                or_(
                    ContactModel.first_name.ilike(search),
                    ContactModel.last_name.ilike(search),
                    ContactModel.display_name.ilike(search),
                )
            )
            .order_by(
                ContactModel.first_name,
                ContactModel.last_name,
            )
        )

        models = self.session.scalars(statement).all()
        return [ContactMapper.to_domain(m) for m in models]

    def find_by_email(
        self,
        email: str,
    ) -> Contact | None:

        statement = (
            select(ContactModel)
            .where(
                ContactModel.email.ilike(email)
            )
        )

        model = self.session.scalar(statement)
        return ContactMapper.to_domain(model) if model else None

    def find_by_phone(
        self,
        phone_number: str,
    ) -> Contact | None:

        statement = (
            select(ContactModel)
            .where(
                ContactModel.phone_number == phone_number
            )
        )

        model = self.session.scalar(statement)
        return ContactMapper.to_domain(model) if model else None

    def find_by_nationality(
        self,
        nationality: str,
    ) -> list[Contact]:

        statement = (
            select(ContactModel)
            .where(
                ContactModel.nationality.ilike(
                    nationality
                )
            )
            .order_by(ContactModel.last_name)
        )

        models = self.session.scalars(statement).all()
        return [ContactMapper.to_domain(m) for m in models]

    def create(
        self,
        contact: Contact,
    ) -> Contact:
        model = ContactMapper.to_model(contact)
        self.session.add(model)
        self.session.flush()
        self.session.commit()

        return ContactMapper.to_domain(model)

    def update(
        self,
        contact: Contact,
    ) -> Contact:
        model = self.session.get(ContactModel, contact.id)
        if model is None:
            raise ValueError(f"Contact {contact.id} not found")

        ContactMapper.update_model(model, contact)
        self.session.flush()
        self.session.commit()

        return ContactMapper.to_domain(model)

    def delete(
        self,
        contact_id: UUID,
    ) -> None:

        model = self.session.get(ContactModel, contact_id)

        if model is not None:
            self.session.delete(model)
            self.session.flush()
            self.session.commit()