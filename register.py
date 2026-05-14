import os
import requests
from typing import Tuple

REQUEST_URL = "https://api.openverse.org/v1/auth_tokens/register/"

def register(name: str, description: str, email: str) -> Tuple[str, str]:
    payload = {
        "name": name,
        "description": description,
        "email": email,
    }

    r = requests.post(
        REQUEST_URL,
        json=payload,
        headers={"User-Agent": "meat-doneness-dataset/1.0"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    return data["client_id"], data["client_secret"]


def write_env(client_id: str, client_secret: str, path: str = ".env"):
    if os.path.exists(path):
        raise RuntimeError(
            f"{path} already exists. Refusing to overwrite secrets."
        )

    with open(path, "w", encoding="utf-8") as env:
        env.write(f'OPENVERSE_CLIENT_ID="{client_id}"\n')
        env.write(f'OPENVERSE_CLIENT_SECRET="{client_secret}"\n')


if __name__ == "__main__":
    client_id, client_secret = register(
        "Meat Doneness Image Dataset",
        """
        I am collecting openly licensed images of cooked meat (primarily beef steaks)
        in clearly identifiable doneness states (raw, rare, medium rare, medium, well done).

        The dataset will be used to train and evaluate a computer vision / machine learning
        model that predicts meat doneness from an image.

        This is a non-commercial academic project for my university.
        All images will be downloaded with full license metadata preserved
        and used in accordance with their respective Creative Commons or Public Domain licenses.
        """,
        "leo@richybaby.com",
    )

    print("Client ID:", client_id)
    print("Client secret received (store it safely!) :", client_secret)

    write_env(client_id, client_secret)
