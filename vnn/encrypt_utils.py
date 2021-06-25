from secureprotol.paillier import PaillierEncrypt


class PaillierEncryptHelper(object):
    def __init__(self, key_length=1024, close_encrypt=False):
        super(PaillierEncryptHelper, self).__init__()
        self.encrypter = PaillierEncrypt()
        self.encrypter.generate_key(key_length)
        self.close_encrypt = close_encrypt

    def encrypt(self, input):
        return input if self.close_encrypt else self.encrypter.recursive_encrypt(input)

    def decrypt(self, input):
        return input if self.close_encrypt else self.encrypter.recursive_decrypt(input)
