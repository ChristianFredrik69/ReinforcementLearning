from Card import Card as Card

import random as rnd

class CardDeck:
    
    def __init__(self):
        
        self.deck = []
        self.build()
        
    def build(self):
        
        """
        Builds a new deck with all the 108 cards, and shuffles the deck.
        """
        
        card_info = Card.info
        for color in card_info["color"]:
            
            #All the cards apart from the wilds
            for num in card_info["trait"][:13]:
                self.deck.append(Card(color, num))
                if num != "0":
                    self.deck.append(Card(color, num))
                
            #The wild cards    
            for wild in card_info["trait"][-2:]:
                self.deck.append(Card(color,wild))
        
        self.shuffle()



    def draw_top_card(self, discard_pile):
        """
        Draws the top card of the deck.
        If there are no cards left in the deck after a draw, the deck gets shuffled.
        """

        top_card = self.deck.pop()

        if self.get_deck_length == 0:
            cards_to_shuffle = discard_pile[:-1]
            discard_pile = discard_pile[-1:]
            self.deck = rnd.shuffle(cards_to_shuffle)

        return top_card
    
    def return_card(self, card):
        """
        Returns a card to the deck.
        """
        self.deck.append(card)
    
    def shuffle(self):
        """
        Shuffles the deck.
        """
        rnd.shuffle(self.deck)
        
    def print_deck(self):
        """
        Prints the whole deck to terminal
        """
        
        for card in self.deck:
            
            print(card.get_string() + "\n")
    
    def get_deck_length(self):
        """
        Returns the amount of cards left in the deck.
        """
        return len(self.deck)
            
if __name__ == "__main__":
    deck = CardDeck()
    print(deck.deck[0])
    print(deck.deck[0].trait)
    print(deck.deck[0].trait)
    print(deck.deck[0].color)
