class Chatbot:
    def __init__(self):
        self.username = ''
        self.password = ''
        self.logged_in = False
        self.menu()

    def menu(self):
        user_input = input(
            "Welcome to Our Chatbot! How would you like to proceed?\n"
            "1. Press 1 to signup\n"
            "2. Press 2 to signin\n"
            "3. Press 3 to write a post\n"
            "4. Press 4 to message a friend\n"
            "5. Press any other key to exit\n"
            "> "
        )

        if user_input == "1":
            self.signup()
        elif user_input == "2":
            self.signin()
        elif user_input == "3":
            self.my_post()
        elif user_input == "4":
            self.sendmsg()
        else:
            exit()


bot = Chatbot()
