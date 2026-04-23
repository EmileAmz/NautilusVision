import numpy as np


def find_angle_plane(boxes):
    profondeurs = []
    angles = []
    C = 1442

    for box in boxes:
        profondeurs.append(box["depth"])
        angles.append(box["angle"])

    h_3 = np.abs(profondeurs[1] - profondeurs[0])
    angle = np.arccos(h_3/C)
    angle = np.degrees(angle)

    result = 90 - angle

    return result

def find_angle_plane_V2(profondeurs, C):

    if profondeurs[0] == 0 or  profondeurs[1] == 0 or profondeurs[1] is None or profondeurs[0] is None:
        return -1000

    else:
        h_3 = np.abs(profondeurs[1] - profondeurs[0])
        angle = np.arccos(h_3/C)
        angle = np.degrees(angle)

        result = 90 - angle
        return result


def switch_case_sub_angle(results):
    results_angle = []

    # Dictionnaire: clé = id objet, valeur = infos
    objets = {}

    for row in results:
        object_id = int(row[0])
        depth = row[1]
        angle = row[2]

        if object_id not in objets:
            # premier objet de cet ID
            objets[object_id] = {
                "depth": depth,
                "angle": angle
            }
        else:
            # comparer avec celui déjà stocké
            if depth < objets[object_id]["depth"]:
                objets[object_id] = {
                    "depth": depth,
                    "angle": angle
                }
    # Présence des objets
    gate_left = 0 in objets
    gate_right = 1 in objets
    gate_middle = 2 in objets
    slalom_cote = 3 in objets
    slalom_middle = 4 in objets

    # Exemple : angle entre gate_left (0) et gate_middle (2)
    if gate_left and gate_middle:
        profondeurs = [objets[0]["depth"], objets[2]["depth"]]
        angle_calc = find_angle_plane_V2(profondeurs, 1442)

        results_angle.append({
            "id": 1,
            "angle": angle_calc,
            "objets_utilises": [0, 2]
        })

    # Exemple : angle entre gate_right (1) et gate_middle (2)
    if gate_right and gate_middle:
        profondeurs = [objets[1]["depth"], objets[2]["depth"]]
        angle_calc = find_angle_plane_V2(profondeurs, 1442)

        results_angle.append({
            "id": 2,
            "angle": angle_calc,
            "objets_utilises": [1, 2]
        })

    # Exemple : angle entre slalom_cote (3) et slalom_middle (4)
    if slalom_cote and slalom_middle:
        profondeurs = [objets[3]["depth"], objets[4]["depth"]]
        angle_calc = find_angle_plane_V2(profondeurs, 1442)

        results_angle.append({
            "id": 3,
            "angle": angle_calc,
            "objets_utilises": [3, 4]
        })

    return results_angle