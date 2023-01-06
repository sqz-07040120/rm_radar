import converts

def NoConv(buff:bytes):
	return buff

packet_dict = {'float':converts.float24bytes,
               'uint8':converts.int21bytes,'int8':converts.int21bytes,
			   'uint16':converts.int22bytes,'int16':converts.int22bytes,
			   'uint32':converts.int24bytes,'int32':converts.int24bytes,
			   'bytes':NoConv,'string':converts.str2hex}

def packet(*argv):
	if argv.count() % 2 == 1:
		raise IndexError
	buff = bytes()
	count = int()
	for i in range(argv.count()/2):
		if argv[i+2+1] in packet_dict:
			buff = buff + packet_dict[argv[i*2+1]](argv[i*2])
	return buff
