from fastapi import APIRouter, Depends, HTTPException

from app.core.success_response import success_response
from app.schemas.contact import ContactCreate, ContactOut
from app.services.contact_service import contact_service
from app.utils.auth_utils import get_current_user_is_admin


contacts_router = APIRouter(prefix="/api/contacts", tags=["contacts"])


def _require_admin(is_admin: bool):
    if not is_admin:
        raise HTTPException(status_code=403, detail="无权限管理联系人")


@contacts_router.get("")
async def list_contacts(is_admin: bool = Depends(get_current_user_is_admin)):
    _require_admin(is_admin)
    rows = await contact_service.list_contacts()
    return success_response(
        data={"contacts": [ContactOut(**row.__dict__).model_dump() for row in rows]}
    )


@contacts_router.post("")
async def create_contact(
    body: ContactCreate, is_admin: bool = Depends(get_current_user_is_admin)
):
    _require_admin(is_admin)
    row = await contact_service.create_contact(**body.model_dump())
    return success_response(
        data=ContactOut(**row.__dict__).model_dump(), message="联系人已创建"
    )
