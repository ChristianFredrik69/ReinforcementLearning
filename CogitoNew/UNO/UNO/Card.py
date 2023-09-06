

class Card:
    
    info =  {"color": ["r", "g", "b", "y"],
            "trait": ["0","1","2","3","4","5","6","7","8","9", "skip", "reverse", "draw_2", "wild", "wild_draw_4"]}

    def __init__(self, color, trait):
        self.color = color
        self.trait = trait
    

    def get_string(self):
        """
        Returns a string representation of the card. |
        Format) color-trait | 
        Example) r-3 (red 3)
        """    
        return self.color + "-" + self.trait
    
    def __str__(self):
        return self.get_string()