
class Player():


    def __init__(self, player_id):
        self.hand = []
        self.player_id = player_id
        


    def get_player_id(self):
        """
        Number between 0 and 3 which is unique to the player.
        """
        return self.player_id
    

    def add_card(self, card):
        """
        Adds a card to the player's hand
        """
        self.hand.append(card)
        
    def get_hand(self):
        """
        Returns a list with all of the cards the player possesses.
        """
        return self.hand

    def play_card(self, card_as_int, discard_pile):
        """
        Makes a player play the specified card, and adds the card to the discard pile.
        """
        card = self.hand.pop(card_as_int)
        discard_pile.append(card)
        return card
        
    


    






