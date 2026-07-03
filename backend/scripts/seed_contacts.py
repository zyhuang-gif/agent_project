"""P0 手动 seed 联系人。示例：
uv run python -m scripts.seed_contacts --name 张三 --email you@example.com --alias zhangsan
"""
import argparse
import asyncio

from app.services.contact_service import contact_service


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--alias", default="")
    parser.add_argument("--note", default="seeded for MCP smoke test")
    args = parser.parse_args()

    row = await contact_service.create_contact(
        name=args.name,
        email=args.email,
        alias=args.alias,
        note=args.note,
        is_active=True,
    )
    print(f"seeded contact id={row.id} name={row.name} email={row.email}")


if __name__ == "__main__":
    asyncio.run(main())
