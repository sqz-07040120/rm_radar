# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nmessage.proto\x12\rProtobuf_test\"4\n\x0b\x41\x64\x64ressBook\x12%\n\x06people\x18\x01 \x03(\x0b\x32\x15.Protobuf_test.Person\"\xa9\x01\n\x06Person\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\x05\x12\r\n\x05\x65mail\x18\x03 \x01(\t\x12\r\n\x05money\x18\x04 \x01(\x02\x12\x13\n\x0bwork_status\x18\x05 \x01(\x08\x12*\n\x06phones\x18\x06 \x03(\x0b\x32\x1a.Protobuf_test.PhoneNumber\x12&\n\x04maps\x18\x07 \x01(\x0b\x32\x18.Protobuf_test.MyMessage\"E\n\x0bPhoneNumber\x12\x0e\n\x06number\x18\x01 \x01(\t\x12&\n\x04type\x18\x02 \x01(\x0e\x32\x18.Protobuf_test.PhoneType\"v\n\tMyMessage\x12\x38\n\x08mapfield\x18\x01 \x03(\x0b\x32&.Protobuf_test.MyMessage.MapfieldEntry\x1a/\n\rMapfieldEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01*+\n\tPhoneType\x12\n\n\x06MOBILE\x10\x00\x12\x08\n\x04HOME\x10\x01\x12\x08\n\x04WORK\x10\x02\x62\x06proto3')

_PHONETYPE = DESCRIPTOR.enum_types_by_name['PhoneType']
PhoneType = enum_type_wrapper.EnumTypeWrapper(_PHONETYPE)
MOBILE = 0
HOME = 1
WORK = 2


_ADDRESSBOOK = DESCRIPTOR.message_types_by_name['AddressBook']
_PERSON = DESCRIPTOR.message_types_by_name['Person']
_PHONENUMBER = DESCRIPTOR.message_types_by_name['PhoneNumber']
_MYMESSAGE = DESCRIPTOR.message_types_by_name['MyMessage']
_MYMESSAGE_MAPFIELDENTRY = _MYMESSAGE.nested_types_by_name['MapfieldEntry']
AddressBook = _reflection.GeneratedProtocolMessageType('AddressBook', (_message.Message,), {
  'DESCRIPTOR' : _ADDRESSBOOK,
  '__module__' : 'test_pb2'
  # @@protoc_insertion_point(class_scope:Protobuf_test.AddressBook)
  })
_sym_db.RegisterMessage(AddressBook)

Person = _reflection.GeneratedProtocolMessageType('Person', (_message.Message,), {
  'DESCRIPTOR' : _PERSON,
  '__module__' : 'test_pb2'
  # @@protoc_insertion_point(class_scope:Protobuf_test.Person)
  })
_sym_db.RegisterMessage(Person)

PhoneNumber = _reflection.GeneratedProtocolMessageType('PhoneNumber', (_message.Message,), {
  'DESCRIPTOR' : _PHONENUMBER,
  '__module__' : 'test_pb2'
  # @@protoc_insertion_point(class_scope:Protobuf_test.PhoneNumber)
  })
_sym_db.RegisterMessage(PhoneNumber)

MyMessage = _reflection.GeneratedProtocolMessageType('MyMessage', (_message.Message,), {

  'MapfieldEntry' : _reflection.GeneratedProtocolMessageType('MapfieldEntry', (_message.Message,), {
    'DESCRIPTOR' : _MYMESSAGE_MAPFIELDENTRY,
    '__module__' : 'test_pb2'
    # @@protoc_insertion_point(class_scope:Protobuf_test.MyMessage.MapfieldEntry)
    })
  ,
  'DESCRIPTOR' : _MYMESSAGE,
  '__module__' : 'test_pb2'
  # @@protoc_insertion_point(class_scope:Protobuf_test.MyMessage)
  })
_sym_db.RegisterMessage(MyMessage)
_sym_db.RegisterMessage(MyMessage.MapfieldEntry)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MYMESSAGE_MAPFIELDENTRY._options = None
  _MYMESSAGE_MAPFIELDENTRY._serialized_options = b'8\001'
  _PHONETYPE._serialized_start=446
  _PHONETYPE._serialized_end=489
  _ADDRESSBOOK._serialized_start=29
  _ADDRESSBOOK._serialized_end=81
  _PERSON._serialized_start=84
  _PERSON._serialized_end=253
  _PHONENUMBER._serialized_start=255
  _PHONENUMBER._serialized_end=324
  _MYMESSAGE._serialized_start=326
  _MYMESSAGE._serialized_end=444
  _MYMESSAGE_MAPFIELDENTRY._serialized_start=397
  _MYMESSAGE_MAPFIELDENTRY._serialized_end=444
# @@protoc_insertion_point(module_scope)