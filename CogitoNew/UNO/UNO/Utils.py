import numpy as np


actions = {0: "r-0", 1: "r-1", 2: "r-2", 3: "r-3", 4: "r-4", 5: "r-5", 6: "r-6", 7: "r-7", 8: "r-8", 9: "r-9", 10: "r-skip", 11: "r-reverse", 12: "r-draw_2", 13: "r-wild", 14: "r-wild_draw_4", 15: "g-0", 16: "g-1", 17: "g-2", 18: "g-3", 19: "g-4", 20: "g-5", 21: "g-6", 22: "g-7", 23: "g-8", 24: "g-9", 25: "g-skip", 26: "g-reverse", 27: "g-draw_2", 28: "g-wild", 29: "g-wild_draw_4", 30: "b-0", 31: "b-1", 32: "b-2", 33: "b-3", 34: "b-4", 35: "b-5", 36: "b-6", 37: "b-7", 38: "b-8", 39: "b-9", 40: "b-skip", 41: "b-reverse", 42: "b-draw_2", 43: "b-wild", 44: "b-wild_draw_4", 45: "y-0", 46: "y-1", 47: "y-2", 48: "y-3", 49: "y-4", 50: "y-5", 51: "y-6", 52: "y-7", 53: "y-8", 54: "y-9", 55: "y-skip", 56: "y-reverse", 57: "y-draw_2", 58: "y-wild", 59: "y-wild_draw_4", 60: "draw", 61: "pass"}
inverted_actions = {"r-0": 0, "r-1": 1, "r-2": 2, "r-3": 3, "r-4": 4, "r-5": 5, "r-6": 6, "r-7": 7, "r-8": 8, "r-9": 9, "r-skip": 10, "r-reverse": 11, "r-draw_2": 12, "r-wild": 13, "r-wild_draw_4": 14, "g-0": 15, "g-1": 16, "g-2": 17, "g-3": 18, "g-4": 19, "g-5": 20, "g-6": 21, "g-7": 22, "g-8": 23, "g-9": 24, "g-skip": 25, "g-reverse": 26, "g-draw_2": 27, "g-wild": 28, "g-wild_draw_4": 29, "b-0": 30, "b-1": 31, "b-2": 32, "b-3": 33, "b-4": 34, "b-5": 35, "b-6": 36, "b-7": 37, "b-8": 38, "b-9": 39, "b-skip": 40, "b-reverse": 41, "b-draw_2": 42, "b-wild": 43, "b-wild_draw_4": 44, "y-0": 45, "y-1": 46, "y-2": 47, "y-3": 48, "y-4": 49, "y-5": 50, "y-6": 51, "y-7": 52, "y-8": 53, "y-9": 54, "y-skip": 55, "y-reverse": 56, "y-draw_2": 57, "y-wild": 58, "y-wild_draw_4": 59, "draw": 60, "pass": 61}


def cards_to_list(player, discard_pile):

    cards1 = [np.zeros(15), np.zeros(15), np.zeros(15), np.zeros(15)]
    cards2 = [np.zeros(15), np.zeros(15), np.zeros(15), np.zeros(15)]
    no_cards = [np.zeros(15), np.zeros(15), np.zeros(15), np.zeros(15)]
    top_pile_card = [np.zeros(15), np.zeros(15), np.zeros(15), np.zeros(15)]

    top_card = discard_pile[-1]
    top_card.index()
    top_pile_card

    cards = [str(card) for card in player.get_hand()]


    for card in cards:
        if (cards.count(card) == 2):
            index = inverted_actions.get(card)
        


    for card in inverted_actions.keys():
         if cards
