"""Turn a title into a URL slug."""


def slugify(text):
    return text.lower().replace(" ", "-")
