def proto_names(pt):
    return [d.name for d in pt.DESCRIPTOR.fields]

def print_pb_stats(obj):
    #From Leela
    for descriptor in obj.DESCRIPTOR.fields:
        value = getattr(obj, descriptor.name)
        if descriptor.name == "weights": return
        if descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                map(print_pb_stats, value)
            else:
                print_pb_stats(value)
        elif descriptor.type == descriptor.TYPE_ENUM:
            enum_name = descriptor.enum_type.values[value].name
            print("%s: %s" % (descriptor.full_name, enum_name))
        else:
            print("%s: %s" % (descriptor.full_name, value))

def print_game(pr_game):
    game_infos = []
    for descriptor in pr_game.DESCRIPTOR.fields:
        name = descriptor.name
        if name != 'boards':
            game_infos.append(f"{descriptor.name}: {getattr(pr_game, name)}")
    print(f"Game: {', '.join(game_infos)}")
    for board in pr_game.boards:
        print_board(board)

def print_board(pr_board):
    board_infos = []
    for descriptor in pr_board.DESCRIPTOR.fields:
        name = descriptor.name
        if name not in ['tree', 'fen']:
            board_infos.append(f"{descriptor.name}: {getattr(pr_board, name)}")
    print(f"Board: {', '.join(board_infos)}")
    print(f"       fen: {pr_board.fen}")
    print_tree(pr_board.tree, prefix = '       ', move_index = pr_board.move_index)

def print_tree(pr_tree, prefix = '', move_index = None):
    tree_infos = []
    for descriptor in pr_tree.DESCRIPTOR.fields:
        name = descriptor.name
        if 'child' not in name:
            tree_infos.append(f"{descriptor.name}: {getattr(pr_tree, name)}")
    tree_infos.append(f"num_info_child: {len(pr_tree.child_moves)}")
    tree_infos.append(f"num_child: {len(pr_tree.children)}")


    print(f"{prefix}Tree: {', '.join(tree_infos)}", end = '')
    move_vals = []
    for m, v in zip(pr_tree.child_moves, pr_tree.child_values):
        move_vals.append(f"{m}: {v:.1f}")

    for i, s in enumerate(move_vals):
        if i % 4 == 0:
            print(f"\n{prefix} \t", end = '')
        if move_index is not None and i == move_index:
            print('\033[92m', end = '')
        print(f"{i+1}. {s},".ljust(15), end = ' ')
        if move_index is not None and i == move_index:
            print('\033[0m', end = '')
    print()
