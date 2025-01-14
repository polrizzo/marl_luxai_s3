import wandb
import random
import datetime

date_now = datetime.datetime.now()
name_exp = "exp_" + date_now.strftime("%Y_%m_%d_%H_%M_%S")

wandb.init(
    entity="polrizzo",
    project="LuxAI_S3",
    dir="./",
    # id: (str | None) = None,
    name="test_wandb_local",
    config={},
    # group: (str | None) = None,
    # job_type: (str | None) = None,
    # reinit: (bool | None) = None,
    # resume: (bool | Literal['allow', 'never', 'must', 'auto'] | None) = None,
    # resume_from: (str | None) = None,
    # fork_from: (str | None) = None,
    # save_code: (bool | None) = None,
    # settings: (Settings | dict[str, Any] | None) = None
)

for epoch in range(2, 100):
    offset = random.random() / 5
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})


wandb.finish()