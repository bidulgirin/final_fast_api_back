import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

KEY = b"12345678901234567890123456789012"  # Android랑 동일해야 함

def decrypt_aes(iv_b64: str, encrypted: bytes) -> bytes:
    iv = base64.b64decode(iv_b64)

    if len(iv) != 16:
        raise ValueError(f"Invalid IV length: {len(iv)} (expected 16)")

    if len(encrypted) % 16 != 0:
        raise ValueError(f"Ciphertext length not multiple of 16: {len(encrypted)}")

    cipher = Cipher(algorithms.AES(KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_plain = decryptor.update(encrypted) + decryptor.finalize()
    
    
    # PKCS5Padding == PKCS7Padding(128)
    unpadder = padding.PKCS7(128).unpadder()
    plain = unpadder.update(padded_plain) + unpadder.finalize()
    return plain
