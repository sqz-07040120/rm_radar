import converts
import serial_port

refer_data_dict = dict()


def ColorNum2ID(num, color):
    if color == 'red':
        return num
    if color == 'blue':
        return num + 100
    return 0


node_ID = {'sentry': 7, 'radar': 9, 'air': 6, 'infantry_one': 3,
           'infantry_two': 4, 'infantry_three': 5, 'engineer': 2, 'hero': 1}


def __game_robot_status_t(buff: bytes):
    if buff[0] > 100:
        refer_data_dict['color'] = 'blue'
        refer_data_dict['id'] = buff[0] - 100
    else:
        refer_data_dict['color'] = 'red'
        refer_data_dict['id'] = buff[0]


def __game_robot_HP_t(buff: bytes):
    refer_data_dict['red_1_hp'] = converts.hex2uint16(buff[0:2])
    refer_data_dict['red_2_hp'] = converts.hex2uint16(buff[2:4])
    refer_data_dict['red_3_hp'] = converts.hex2uint16(buff[4:6])
    refer_data_dict['red_4_hp'] = converts.hex2uint16(buff[6:8])
    refer_data_dict['red_5_hp'] = converts.hex2uint16(buff[8:10])
    refer_data_dict['red_7_hp'] = converts.hex2uint16(buff[10:12])
    refer_data_dict['red_outpost_hp'] = converts.hex2uint16(buff[12:14])
    refer_data_dict['red_base_hp'] = converts.hex2uint16(buff[14:16])
    refer_data_dict['blue_1_hp'] = converts.hex2uint16(buff[16:18])
    refer_data_dict['blue_2_hp'] = converts.hex2uint16(buff[18:20])
    refer_data_dict['blue_3_hp'] = converts.hex2uint16(buff[20:22])
    refer_data_dict['blue_4_hp'] = converts.hex2uint16(buff[22:24])
    refer_data_dict['blue_5_hp'] = converts.hex2uint16(buff[24:26])
    refer_data_dict['blue_7_hp'] = converts.hex2uint16(buff[26:28])
    refer_data_dict['blue_outpost_hp'] = converts.hex2uint16(buff[28:30])
    refer_data_dict['blue_base_hp'] = converts.hex2uint16(buff[30:32])


def __event_data_t(buff: bytes):
    refer_data_dict['event'] = converts.hex2uint32(buff)


def __dart_cmd_t(buff: bytes):
    refer_data_dict['dart_launch_opening_status'] = converts.hex2uint8(buff[0])
    refer_data_dict['dart_attack_target'] = converts.hex2uint8(buff[1])


def __robot_interactive_t(buff: bytes):
    pass


def referee_decode(buff: bytes):
    for key, item in referee_decode_dict:
        if buff[0] == key[1] and buff[1] == key[0]:
            item(buff[2:])


referee_decode_dict = {b'\x02\x01': __game_robot_status_t,
                       b'\x00\x03': __game_robot_HP_t,
                       b'\x01\x01': __event_data_t,
                       b'\x02\x0A': __dart_cmd_t,
                       b'\x03\x01': __robot_interactive_t}


class referee:
    def __init__(self, sp: str) -> None:
        self.decode_func = referee_decode
        self.sp = serial_port.serial_port(
            port=sp, receive_callback_func=self.decode_func)

    def get_a_value(self, name: str):
        return refer_data_dict[name.lower()]

    def send_to_robot(self, node_name: str, cmd_id: bytes, data: bytes):
        bytes_buff_cmdid1 = converts.int22bytes(0x0301)
        bytes_buff_cmdid2 = converts.int22bytes(cmd_id)
        bytes_buff_receiver = converts.int22bytes(
            ColorNum2ID(node_ID[node_name], self.get_a_value('color')))
        bytes_buff_transmiter = converts.int22bytes(
            ColorNum2ID(node_ID['radar'], self.get_a_value('color')))

        self.sp.write_tx(bytes_buff_cmdid1 + bytes_buff_cmdid2 +
                         bytes_buff_receiver + bytes_buff_transmiter + data)

    def draw_map(self, _num, _X, _Y):
        bytes_buff_cmdid = converts.int24bytes(0x0305)
        bytes_buff_num = converts.int22bytes(_num)
        bytes_buff_X = converts.float24bytes(_X)
        bytes_buff_Y = converts.float24bytes(_Y)
        reverse = converts.float24bytes(0)
        self.sp.write_tx(bytes_buff_cmdid + bytes_buff_num +
                         bytes_buff_X + bytes_buff_Y + reverse)
