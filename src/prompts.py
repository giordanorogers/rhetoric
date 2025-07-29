from src.config import RHETORIC

def compose_prompt(
    stmt,
    base_template=None,
    rhetoric_class=None,
    intensity=-1
):
    if base_template is None:
        base_template = "{}\nTrue or False?\nAnswer:"
    if rhetoric_class is None:
        return base_template.format(stmt)
    else:
        prefix = RHETORIC[rhetoric_class][intensity]
        return prefix + "\n" + base_template.format(stmt)