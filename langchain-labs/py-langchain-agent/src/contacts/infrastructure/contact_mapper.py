
from src.contacts.domain.contact import Contact
from src.contacts.models.contact_model import ContactModel


class ContactMapper:

    @staticmethod
    def to_domain(model: ContactModel) -> Contact:
        return Contact(
            id=model.id,
            first_name=model.first_name,
            last_name=model.last_name,
            display_name=model.display_name,
            birthday=model.birthday,
            nationality=model.nationality,
            email=model.email,
            phone_number=model.phone_number,
            address_line_1=model.address_line_1,
            address_line_2=model.address_line_2,
            city=model.city,
            country=model.country,
            company=model.company,
            job_title=model.job_title,
            social_networks=model.social_networks,
            preferences=model.preferences,
            notes=model.notes,
            metadata=model.metadata_,
        )

    @staticmethod
    def to_model(contact: Contact) -> ContactModel:
        return ContactModel(
            id=contact.id,
            first_name=contact.first_name,
            last_name=contact.last_name,
            display_name=contact.display_name,
            birthday=contact.birthday,
            nationality=contact.nationality,
            email=contact.email,
            phone_number=contact.phone_number,
            address_line_1=contact.address_line_1,
            address_line_2=contact.address_line_2,
            city=contact.city,
            country=contact.country,
            company=contact.company,
            job_title=contact.job_title,
            social_networks=contact.social_networks,
            preferences=contact.preferences,
            notes=contact.notes,
            metadata_=contact.metadata,
        )

    @staticmethod
    def update_model(model: ContactModel, contact: Contact) -> ContactModel:
        """Apply domain entity values onto an existing tracked model instance
        (avoids detaching/re-adding an already-persistent row)."""
        model.first_name = contact.first_name
        model.last_name = contact.last_name
        model.display_name = contact.display_name
        model.birthday = contact.birthday
        model.nationality = contact.nationality
        model.email = contact.email
        model.phone_number = contact.phone_number
        model.address_line_1 = contact.address_line_1
        model.address_line_2 = contact.address_line_2
        model.city = contact.city
        model.country = contact.country
        model.company = contact.company
        model.job_title = contact.job_title
        model.social_networks = contact.social_networks
        model.preferences = contact.preferences
        model.notes = contact.notes
        model.metadata_ = contact.metadata
        return model