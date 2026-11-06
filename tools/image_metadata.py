#!/usr/bin/env python3
"""Remove identifying image metadata without recompressing image data.

Keep JPEG orientation and color profiles: deleting orientation would rotate
phone photos in browsers. Never copy the EXIF thumbnail, GPS, camera serial,
timestamps, XMP, IPTC, or free-text comments into a published image.
"""

import argparse
import struct
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def orientation_exif(value):
    if value not in range(2, 9):
        return b""
    return b"Exif\x00\x00" + struct.pack(
        "<2sHIHHHIII", b"II", 42, 8, 1, 274, 3, 1, value, 0
    )


def read_orientation(exif):
    """Read only IFD0's scalar orientation; do not traverse identifying IFDs."""
    if not exif.startswith(b"Exif\x00\x00"):
        return 1
    tiff = exif[6:]
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        raise ValueError("invalid EXIF header")
    endian = "<" if tiff[:2] == b"II" else ">"
    magic, offset = struct.unpack_from(endian + "HI", tiff, 2)
    if magic != 42 or offset + 2 > len(tiff):
        raise ValueError("invalid EXIF directory")
    count = struct.unpack_from(endian + "H", tiff, offset)[0]
    for index in range(count):
        pos = offset + 2 + index * 12
        if pos + 12 > len(tiff):
            raise ValueError("truncated EXIF directory")
        tag, kind, length = struct.unpack_from(endian + "HHI", tiff, pos)
        if tag == 274:
            if kind != 3 or length != 1:
                raise ValueError("invalid EXIF orientation")
            value = struct.unpack_from(endian + "H", tiff, pos + 8)[0]
            if value not in range(1, 9):
                raise ValueError("invalid EXIF orientation")
            return value
    return 1


def clean_jpeg(data):
    out = bytearray(data[:2])
    pos = 2
    orientation_written = False
    while pos < len(data):
        start = pos
        if data[pos] != 0xFF:
            raise ValueError("invalid JPEG marker")
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos == len(data):
            raise ValueError("truncated JPEG marker")
        marker = data[pos]
        pos += 1
        if marker == 0xD9:
            out.extend(data[start:pos])
            return bytes(out)  # Do not retain bytes appended after the image.
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            out.extend(data[start:pos])
            continue
        if pos + 2 > len(data):
            raise ValueError("truncated JPEG segment")
        size = int.from_bytes(data[pos:pos + 2], "big")
        if size < 2 or pos + size > len(data):
            raise ValueError("invalid JPEG segment size")
        payload = data[pos + 2:pos + size]
        end = pos + size
        # All non-rendering APP records are removed. Retain JFIF/JFXX, ICC
        # profiles and Adobe color-transform information, which affect display.
        keep = not (0xE0 <= marker <= 0xEF or marker == 0xFE)
        keep |= marker == 0xE0 and payload.startswith((b"JFIF\x00", b"JFXX\x00"))
        keep |= marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00")
        keep |= marker == 0xEE and payload.startswith(b"Adobe")
        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            orientation = read_orientation(payload)
            if not orientation_written:
                safe = orientation_exif(orientation)
                if safe:
                    out.extend(b"\xff\xe1" + struct.pack(">H", len(safe) + 2) + safe)
                orientation_written = True
        if keep:
            out.extend(data[start:end])
        pos = end
        if marker == 0xDA:
            # Copy entropy-coded pixels verbatim, including escaped FF bytes
            # and restart markers. Progressive images can have several scans.
            scan_start = pos
            while pos < len(data):
                if data[pos] != 0xFF:
                    pos += 1
                    continue
                next_pos = pos + 1
                while next_pos < len(data) and data[next_pos] == 0xFF:
                    next_pos += 1
                if next_pos == len(data):
                    raise ValueError("truncated JPEG scan")
                if data[next_pos] == 0 or 0xD0 <= data[next_pos] <= 0xD7:
                    pos = next_pos + 1
                    continue
                out.extend(data[scan_start:pos])
                break
    raise ValueError("JPEG has no end marker")


def clean_png(data):
    out = bytearray(PNG_SIGNATURE)
    pos = len(PNG_SIGNATURE)
    while pos + 12 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        end = pos + length + 12
        if end > len(data):
            raise ValueError("truncated PNG chunk")
        kind = data[pos + 4:pos + 8]
        if kind not in (b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"tIME"):
            out.extend(data[pos:end])
        if kind == b"IEND":
            return bytes(out)
        pos = end
    raise ValueError("PNG has no end chunk")


def clean_image(data):
    if data.startswith(b"\xff\xd8"):
        return clean_jpeg(data)
    if data.startswith(PNG_SIGNATURE):
        return clean_png(data)
    if data.startswith((b"GIF87a", b"GIF89a")):
        return clean_gif(data)
    raise ValueError("expected a JPEG, PNG, or GIF")


def clean_gif(data):
    if len(data) < 13:
        raise ValueError("truncated GIF header")
    pos = 13 + (3 * (2 << (data[10] & 7)) if data[10] & 128 else 0)
    out = bytearray(data[:pos])
    while pos < len(data):
        start = pos
        marker = data[pos]
        pos += 1
        if marker == 0x3B:
            return bytes(out) + b";"
        keep = True
        if marker == 0x21:
            if pos >= len(data):
                raise ValueError("truncated GIF extension")
            label = data[pos]
            pos += 1
            # Preserve animation loops and rendering controls; discard comment,
            # XMP and unknown application extensions that can carry free text.
            keep = label == 0xF9 or (
                label == 0xFF and data[pos:pos + 12] in
                (b"\x0bNETSCAPE2.0", b"\x0bANIMEXTS1.0")
            )
        elif marker == 0x2C:
            if pos + 9 > len(data):
                raise ValueError("truncated GIF image")
            packed = data[pos + 8]
            pos += 9 + (3 * (2 << (packed & 7)) if packed & 128 else 0)
            pos += 1  # LZW minimum code size
        else:
            raise ValueError("invalid GIF block")
        while pos < len(data):
            size = data[pos]
            pos += 1
            if pos + size > len(data):
                raise ValueError("truncated GIF sub-block")
            pos += size
            if size == 0:
                break
        else:
            raise ValueError("unterminated GIF sub-block")
        if keep:
            out.extend(data[start:pos])
    raise ValueError("GIF has no trailer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true", help="strip metadata in place")
    args = parser.parse_args()
    changed = 0
    for path in args.paths:
        original = path.read_bytes()
        cleaned = clean_image(original)
        if original != cleaned:
            changed += 1
            print(f"{path}: identifying metadata {'removed' if args.write else 'found'}")
            if args.write:
                path.write_bytes(cleaned)
    return 0 if args.write or not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
