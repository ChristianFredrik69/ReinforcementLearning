from CardDeck import CardDeck as Deck
from Player import Player
from Card import Card
import re
import numpy as np

class Game():

    actions = {0: "r-0", 1: "r-1", 2: "r-2", 3: "r-3", 4: "r-4", 5: "r-5", 6: "r-6", 7: "r-7", 8: "r-8", 9: "r-9", 10: "r-skip", 11: "r-reverse", 12: "r-draw_2", 13: "r-wild", 14: "r-wild_draw_4", 15: "g-0", 16: "g-1", 17: "g-2", 18: "g-3", 19: "g-4", 20: "g-5", 21: "g-6", 22: "g-7", 23: "g-8", 24: "g-9", 25: "g-skip", 26: "g-reverse", 27: "g-draw_2", 28: "g-wild", 29: "g-wild_draw_4", 30: "b-0", 31: "b-1", 32: "b-2", 33: "b-3", 34: "b-4", 35: "b-5", 36: "b-6", 37: "b-7", 38: "b-8", 39: "b-9", 40: "b-skip", 41: "b-reverse", 42: "b-draw_2", 43: "b-wild", 44: "b-wild_draw_4", 45: "y-0", 46: "y-1", 47: "y-2", 48: "y-3", 49: "y-4", 50: "y-5", 51: "y-6", 52: "y-7", 53: "y-8", 54: "y-9", 55: "y-skip", 56: "y-reverse", 57: "y-draw_2", 58: "y-wild", 59: "y-wild_draw_4", 60: "draw", 61: "pass"}
    inverted_actions = {"r-0": 0, "r-1": 1, "r-2": 2, "r-3": 3, "r-4": 4, "r-5": 5, "r-6": 6, "r-7": 7, "r-8": 8, "r-9": 9, "r-skip": 10, "r-reverse": 11, "r-draw_2": 12, "r-wild": 13, "r-wild_draw_4": 14, "g-0": 15, "g-1": 16, "g-2": 17, "g-3": 18, "g-4": 19, "g-5": 20, "g-6": 21, "g-7": 22, "g-8": 23, "g-9": 24, "g-skip": 25, "g-reverse": 26, "g-draw_2": 27, "g-wild": 28, "g-wild_draw_4": 29, "b-0": 30, "b-1": 31, "b-2": 32, "b-3": 33, "b-4": 34, "b-5": 35, "b-6": 36, "b-7": 37, "b-8": 38, "b-9": 39, "b-skip": 40, "b-reverse": 41, "b-draw_2": 42, "b-wild": 43, "b-wild_draw_4": 44, "y-0": 45, "y-1": 46, "y-2": 47, "y-3": 48, "y-4": 49, "y-5": 50, "y-6": 51, "y-7": 52, "y-8": 53, "y-9": 54, "y-skip": 55, "y-reverse": 56, "y-draw_2": 57, "y-wild": 58, "y-wild_draw_4": 59, "draw": 60, "pass": 61}

    def __init__(self, num_players):

        self.players = []
        self.cardDeck = Deck()
        self.discardPile = []
        self.current_player = 0
        self.card_drawn_flag = False
        self.rotation = False
        
        self.init_players_deal_cards(num_players)
        self.draw_first_discard_pile_card()
        
        

    def init_players_deal_cards(self, num_players):

        """
        Creates the specified number of players, and deals cards 7 cards to each of them.
        """

        for i in range(num_players):
            self.players.append(Player(i))
        
        for i in range(num_players):
            for j in range(7):
                self.players[i].add_card(self.cardDeck.draw_top_card(self.discardPile))

    def draw_first_discard_pile_card(self):
        """
        There is a rule which says that the first card cannot be a draw-4.
        Also, the effects of all other cards should apply.
        """
        drawn_card = self.cardDeck.draw_top_card(self.discardPile)


        while (True):
            if drawn_card.trait == "wild_draw_4":
                self.cardDeck.return_card(drawn_card)
                self.cardDeck.shuffle()
                drawn_card = self.cardDeck.draw_top_card(self.discardPile)
                continue
            self.discardPile.append(drawn_card)
            break

        self.process_played_card()

    
    def get_next_player(self):
        """
        This method is used to find the player who is next in line.
        """

        num_players = len(self.players)

        if self.rotation:
            self.current_player -= 1
            if self.current_player < 0:
                self.current_player = num_players - 1
        else:
            self.current_player += 1
            if self.current_player >= num_players:
                self.current_player = 0
        
        return self.players[self.current_player]

    def process_played_card(self):
        """
        Handles all of the stuff which happens when a special card is played.
        """
        pattern = re.compile(r'^[0-9]{1}$')
        top_card = self.discardPile[-1]
        text = top_card.trait

        if pattern.match(text):
            return

        elif (top_card.trait == "skip"):
            self.get_next_player()
        
        elif (top_card.trait == "reverse"):
            self.rotation = not self.rotation

        elif (top_card.trait == "draw_2"):
            self.get_next_player()
            for i in range(2):
                self.player_draw_card(self.current_player)


        elif (top_card.trait == "wild"):
            self.wild_card_played()
        
        elif (top_card.trait == "wild_draw_4"):
            self.wild_card_played()
            self.get_next_player()
            for i in range(4):
                self.player_draw_card(self.current_player)

    def is_valid_play(self, card):
        
        """
        Checks if a card can be played.
        """
        
        top_card = self.discardPile[-1]
        color = card.color
        trait = card.trait

        if (trait == "wild" or trait == "wild_draw_4"):
            return True
        
        elif (color == top_card.color):
            return True

        elif (trait == top_card.trait):
            return True
        
        return False

    def wild_card_played(self):
        """
        Makes the player able to choose which color they want to switch to.
        """

        top_card = self.discardPile[-1]

        print("You played a Wild card. Choose a color to continue:")
        print("1. Red\n2. Blue\n3. Green\n4. Yellow")
        
        choice = 0
        while (choice < 1 or choice > 4):
            choice = Game.int_input()

        color_map = {1: "r", 2: "b", 3: "g", 4: "y"}
        top_card.color = color_map[choice]


    def player_draw_card(self, player_id):
        """
        Makes the player draw the top card from the deck.
        """
        card = self.cardDeck.draw_top_card(self.discardPile)
        player = self.players[player_id]
        player.add_card(card)
        return card

    def get_valid_cards(self, player_id):
        """
        Gets a list of the cards the player can play.
        """
        player = self.players[player_id]
        valid_cards = []
        
        for card in player.get_hand():
            if self.is_valid_play(card):
                valid_cards.append(card)
        
        return valid_cards


    def step(self, player_id, action_number):
        """
        Bots use this method to train.
        """
        player = self.players[player_id]
        action = Game.actions.get(action_number)

        if action == "draw":
            drawn_card = self.player_draw_card(self.current_player)
            self.card_drawn_flag = True
            
            if self.is_valid_play(drawn_card):
                string = str(drawn_card)
                legal_actions = np.zeros(62)
                legal_actions[61] = 1
                legal_actions[Game.inverted_actions[string]] = 1

            
        elif action != "pass":
            string = str(drawn_card)
            card = Card(string[0], string[2:])
            player.play_card(player.get_hand().index(card), self.discardPile)
            self.process_played_card()
        
        
        state = 1
        reward = 2

            


    def play_turn(self, player_id):
        
        """
        Handle a single turn for a player.
        """

        player = self.players[player_id]

        if (self.card_drawn_flag):
            
            self.card_drawn_flag = False
            print(str(self.current_player) + "'s hand: " + ", ".join([str(card) for card in player.get_hand()]))
            
            if self.is_valid_play(player.get_hand()[-1]):
                
                print ("You have the following options: ", end=" ")
                print("0: pass", end="  ")
                print("1: " + str(player.get_hand()[-1]), end="    ")
                print("Pick your action: ")

                choice = -1
                while (choice < 0 or choice > 1):
                    choice = Game.int_input()
                
                if choice == 0:
                    return
                
                else:
                    player.play_card(len(player.get_hand()) - 1, self.discardPile)
                    self.process_played_card()
                    return

            else:
                print("No valid moves, enter 0 to continue.")
                choice = -1
                while (choice != 0):
                    choice = Game.int_input()
                return

        valid_cards = self.get_valid_cards(self.current_player)

        if (valid_cards):
            print(str(self.current_player) + "'s hand: " + ", ".join([str(card) for card in player.get_hand()]))
            print ("You have the following options:", end = " ")
            print("0: draw card", end="  ")
            
            for i in range (1, len(valid_cards) + 1):
                print(str(i) + ": " + valid_cards[i-1].get_string(), end="    ")
            
            print("Pick your action: ")
            choice = -1
            while (choice < 0 or choice > len(valid_cards)):
                choice = Game.int_input()
            
            if choice == 0:
                self.player_draw_card(self.current_player)
                self.card_drawn_flag = True
                self.play_turn(self.current_player)
            
            else:
                card = valid_cards[choice - 1]
                player.play_card(player.get_hand().index(card), self.discardPile)
                self.process_played_card()
                return
        
        
        else:
            print("No valid moves, you must draw a card. Enter 0 to continue.")
            choice = -1
            while (choice != 0):
                choice = Game.int_input()
            self.player_draw_card(self.current_player)
            self.card_drawn_flag = True
            self.play_turn(self.current_player)

        
        
    def is_winner(self):
        """
        Checks if the current player has won the game.
        """
        for i in range (len(self.players)):
            if len(self.players[i].get_hand()) == 0:
                return True
        
        return False



    

    
    def play_game(self):
        """
        This method is used to play a game of uno.
        """

        game_over = False
        
        while not game_over:
            
            print("Latest card: " + self.discardPile[-1].get_string())
            self.play_turn(self.current_player)
        
            if self.is_winner():
                print(f"Player {self.current_player} won the game!")
                game_over = True
            else:
                self.get_next_player()


    @staticmethod
    def int_input():
        """
        This method is used to validate inputs.
        """
        input_str = input("Enter a number: ")

        while True:
            if input_str.isdigit():
                choice = int(input_str)
                print("You entered:", choice)
                return choice

            else:
                print("Invalid input. Please enter a valid number.")
                input_str = input("Enter a number: ")
    
if __name__ == "__main__":
    print(np.zeros(3))
    num_players = 4
    game = Game(num_players)
    game.play_game()
