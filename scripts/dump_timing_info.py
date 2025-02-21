import json
import pathlib
import pprint
import npc_sessions

session_id = '715710_2024-07-17'
s = npc_sessions.Session(session_id) 
s.probe_letters_to_use = ('A', 'B', 'C', 'D', 'E', 'F')
d =   {
        i.device.name.removesuffix('-AP').split('.')[-1]: {'start_time':i.start_time, 'sampling_rate':i.sampling_rate} 
        for i in s.ephys_timing_data
        if '-AP' in i.device.name

    }
pprint.pprint(d)
pathlib.Path(f"{session_id}_timing.json").write_text(json.dumps(d))