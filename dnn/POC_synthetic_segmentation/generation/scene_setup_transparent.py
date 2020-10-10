'''
This script is to be used in combination with model files and resource files available at: 
https://uillinoisedu-my.sharepoint.com/personal/yixiaos3_illinois_edu/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly91aWxsaW5vaXNlZHUtbXkuc2hhcmVwb2ludC5jb20vOmY6L2cvcGVyc29uYWwveWl4aWFvczNfaWxsaW5vaXNfZWR1L0VrUWVuVVhWaWVsTWt1a2p1M1dGcjVjQm1veHBGRFFlQUZZek1fZUpFSExBVXc_cnRpbWU9WnQ4TmY2WkgyRWc&id=%2Fpersonal%2Fyixiaos3_illinois_edu%2FDocuments%2FRM 2021%2FAI Vision%2FDataset%2FSynthesis%2FBatch_Render_Infantry
'''
import bpy
import numpy as np
C = bpy.context
D = bpy.data
seed=0
np.random.seed(seed)
print(f'seed: {seed}')

# CHANGE ME
DO_LABEL=True # If we are rendering label or image
DO_NUMBER=False # If numbers are included
NUM_SAMPLE=1e3 # How many images to generate

def get_all_objs():
    objs=[]
    for collection in bpy.data.collections:
       print(collection.name)
       if not 'Exclude' in collection.name:
            for obj in collection.all_objects:
                objs.append(obj)
    return objs

def set_visible_all(objs,visibility=True):
    visibility=not visibility
    for obj in objs:
        obj.hide_set(visibility)
        obj.hide_render=visibility
        obj.hide_viewport=visibility

def update(p1,p2):
    text = bpy.data.objects['t1']
    frame = bpy.context.scene.frame_current
    if DO_NUMBER:
        text.data.body = str(frame%6+1)
    else:
        text.data.body=''


# TODO Lighting, Camera Background
def simulate(scene, robot_objects,label_objects,stage_objects,light_objects,number_objects,obj_camera,camera_target,label=False):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_objs=label_objects+robot_objects+label_objects+stage_objects+light_objects+[obj_camera,camera_target]+number_objects
    for ob in mesh_objs:
        ob.animation_data_clear()
        ob.select_set(state = True)
    if label:
        scene.render.filepath = bpy.data.filepath.rpartition('.')[0]+'/label_out/out_seed{}_'.format(seed)
        obj_camera.data.show_background_images=False
    else:
        scene.render.filepath = bpy.data.filepath.rpartition('.')[0]+'/image_out/out_seed{}_'.format(seed)
        obj_camera.data.show_background_images=True
    set_visible_all(get_all_objs(),not label)
    set_visible_all(label_objects,label)
    
    # Register armor plate text update function, force refresh callback to ensure any modification to update function get represented at new script run
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(update)
    
    scene.frame_set(0)
    for object in mesh_objs:
        object.keyframe_insert('location', frame=0)
        
    for i in range(1, NUM_SAMPLE): 
        # Dataset
        scene.frame_set(i)
        p=4
        phi= np.random.uniform(0.25*np.pi,0.5*np.pi)
        theta =np.random.uniform(0,2*np.pi)
        obj_camera.location.x,obj_camera.location.y,obj_camera.location.z=(p*np.cos(theta)*np.sin(phi),p*np.sin(theta)*np.sin(phi),p*np.cos(phi))
        camera_target.location.x,camera_target.location.y,camera_target.location.z=0,0,0
        obj_camera.keyframe_insert('location', frame=i)
        camera_target.keyframe_insert('location', frame=i)
        
        for obj in stage_objects+light_objects:
            loc = (np.random.random([3])-0.5)*10
            loc[(loc>=0)*(loc<=2)]=2
            loc[(loc>=0)*(loc<=2)]=-2
            obj.location.x,obj.location.y,obj.location.z=tuple(loc)
            obj.keyframe_insert('location', frame=i)
        
        for l in light_objects:
            l.data.node_tree.nodes['Emission'].inputs['Strength'].default_value=np.random.rand()*20
            l.data.node_tree.nodes['Emission'].inputs['Strength'].keyframe_insert(
                data_path='default_value', frame=i)
                
    scene.frame_set(0)



if __name__ == '__main__':
    scene = bpy.context.scene
    obj_camera = bpy.data.objects["Camera"]
    robot_objects = [obj for obj in bpy.context.scene.collection.children.get('Robot').objects]
    label_objects=[obj for obj in bpy.context.scene.collection.children.get('Label').objects]
    stage_objects=[obj for obj in bpy.context.scene.collection.children.get('Environment').objects]
    light_objects=[obj for obj in bpy.context.scene.collection.children.get('Light').objects]
    number_objects=[obj for obj in bpy.context.scene.collection.children.get('Number').objects]
    print(robot_objects)

    simulate(scene, robot_objects,label_objects,stage_objects,light_objects,number_objects,obj_camera,bpy.data.objects["Camera Target"],label=DO_LABEL)
    #bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True) # Enable this line to automatically start rendering after running script
