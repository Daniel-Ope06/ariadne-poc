import bpy
import bmesh
import random

active = bpy.context.active_object

if active.type != 'MESH':  # type: ignore
    raise Exception("Active object must be a mesh object")

assert (active is not None)
assert (active.modifiers is not None)

# Verify active object has maze nodes modifier
maze_nodes = bpy.data.node_groups['Maze']
maze_modifier = None

for modifier in active.modifiers:
    if modifier.type == 'NODES' and modifier.node_group == maze_nodes:
        maze_modifier = modifier
        break

if not maze_modifier:
    maze_modifier = active.modifiers.new('Maze', 'NODES')  # type: ignore
    maze_modifier.node_group = bpy.data.node_groups['Maze']
    for i in range(len(active.modifiers) - 1):  # type: ignore
        bpy.ops.object.modifier_move_up(
            modifier=maze_modifier.name)  # type: ignore


# Verify active mesh has 'closed' attribute
mesh = active.data
bm = bmesh.new()
bm.from_mesh(mesh)  # type: ignore
closed = mesh.attributes.get('closed')  # type: ignore
if not closed:
    closed = mesh.attributes.new(  # type: ignore
        name='closed', type='BOOLEAN', domain='EDGE'
    )
    closed.name = 'closed'


# Reset the maze by closing all edges
for e in bm.edges:
    setattr(closed.data[e.index], 'value', True)

# Randomly chose a cell to start carving the maze from
current_cell = random.choice(list(bm.faces))
visited = [current_cell]
stack = [current_cell]

# Loop until the entire maze is carved out
while len(stack) > 0:
    current_cell = stack.pop()

    # Get unvisited neighbors to current cell
    unvisited_neighbors = []
    for edge in current_cell.edges:
        for neighbor in edge.link_faces:
            if neighbor != current_cell and neighbor not in visited:
                unvisited_neighbors.append(neighbor)

    # Randomly choose an unvisited neighbor
    if len(unvisited_neighbors) > 0:
        stack.append(current_cell)
        chosen_neighbor = random.choice(unvisited_neighbors)
        visited.append(chosen_neighbor)
        stack.append(chosen_neighbor)

        # open all edges between current cell and chosen neighbor
        for edge in current_cell.edges:
            if edge in chosen_neighbor.edges:
                setattr(closed.data[edge.index], 'value', False)

# update the mesh to show the maze
mesh.update()  # type: ignore
