import json

with open("Configure.json", 'r+') as json_file:
    raw_json = json_file.read()
    json_dict = json.loads(raw_json)

def json_parser(json_dict, name):
    for i in range(len(json_dict)):
        json_keys = list(json_dict.keys())
        json_values = list(json_dict.values())
        if type(json_values[i]) == dict:
            subname = name + '_' + str(json_keys[i])
            json_parser(json_values[i], subname)
        else:
            print(name + '_' + str(json_keys[i]) + '_' + str(json_values[i]))



json_parser(json_dict, " ")