import struct


def hex2float(buff: bytes) -> float:
    return struct.unpack('<f', buff)[0]


def hex2uint8(buff: bytes) -> int:
    return struct.unpack('B', buff)[0]


def hex2int8(buff: bytes) -> int:
    return struct.unpack('b', buff)[0]


def hex2uint16(buff: bytes) -> int:
    return struct.unpack('<H', buff)[0]


def hex2int16(buff: bytes) -> int:
    return struct.unpack('<h', buff)[0]


def hex2uint32(buff: bytes) -> int:
    return struct.unpack('<I', buff)[0]


def hex2int32(buff: bytes) -> int:
    return struct.unpack('<i', buff)[0]


def str2hex(buff: str) -> bytes:
    return bytes.fromhex(buff)


def list2str(buff: list) -> str:
    return ''.join(buff)


def list2float(buff: list) -> float:
    return hex2float(str2hex(list2str(buff)))


def list2uint8(buff: list) -> float:
    return hex2uint8(str2hex(list2str(buff)))


def list2int8(buff: list) -> float:
    return hex2int8(str2hex(list2str(buff)))


def list2uint16(buff: list) -> float:
    return hex2uint16(str2hex(list2str(buff)))


def list2int16(buff: list) -> float:
    return hex2int16(str2hex(list2str(buff)))


def list2uint32(buff: list) -> float:
    return hex2uint32(str2hex(list2str(buff)))


def list2int32(buff: list) -> float:
    return hex2uint32(str2hex(list2str(buff)))


def int21bytes(buff: int) -> bytes:
    return int(buff).to_bytes(length=1, byteorder='little')


def int22bytes(buff: int) -> bytes:
    return int(buff).to_bytes(length=2, byteorder='little')


def int24bytes(buff: int) -> bytes:
    return int(buff).to_bytes(length=4, byteorder='little')


def float24bytes(buff: float) -> bytes:
    bs = struct.pack("f", buff)
    return (bs[3], bs[2], bs[1], bs[0])
