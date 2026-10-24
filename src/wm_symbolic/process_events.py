"""All-control-step event targets, including transient device states."""
EVENT_NAMES=('red_moved','blue_moved','pink_moved','led_was_off','led_was_on','light_was_off','light_was_on')
def add_device_events(label,infos):
 values={name:[int(info['scene_info']['lights'][name]['logical_state']) for info in infos] for name in ['led','lightbulb']}
 extra=[int(any(v==state for v in values[name])) for name,state in [('led',0),('led',1),('lightbulb',0),('lightbulb',1)]]
 label=dict(label);label['events']=label['events'][:3]+extra;label['events_valid']=[True]*7;label['event_names']=list(EVENT_NAMES)
 label['schema']='labels_process_v2';label['device_event_first_step']={event:next((i for i,v in enumerate(values[name]) if v==state),None) for event,name,state in [('led_was_off','led',0),('led_was_on','led',1),('light_was_off','lightbulb',0),('light_was_on','lightbulb',1)]}
 return label
