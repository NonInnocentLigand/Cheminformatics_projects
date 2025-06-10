import pandas as pd
import matplotlib.pyplot as plt

classification_type = "multiclass"
model_eval_df = pd.read_csv(f'/Users/leoparsons/Desktop/cheminformatics_projects_drafts/ChEMBL rough drafts/{classification_type}_model.csv')

model_assessment_column_names = ["train_loss", "test_loss", "validation_loss",
                                 "train_accuracy", "test_accuracy", "validation_accuracy"]

model_eval_df.index = model_assessment_column_names
model_eval_df = model_eval_df.T
model_eval_df.reset_index(inplace=True)
model_eval_df.drop('index', axis=1, inplace=True)
model_eval_df.drop(0, inplace=True)

epoch_number = 250
round_value = 3
text_box = f"Performance at Epoch {epoch_number}\n\n" \
           f"training_loss: {round(model_eval_df.loc[epoch_number, 'train_loss'], round_value)}\n" \
           f"test_loss: {round(model_eval_df.loc[epoch_number, 'test_loss'], round_value)}\n" \
           f"validation_loss: {round(model_eval_df.loc[epoch_number, 'validation_loss'], round_value)}\n" \
           f"train_accuracy: {round(model_eval_df.loc[epoch_number, 'train_accuracy'], round_value)}\n" \
           f"test_accuracy: {round(model_eval_df.loc[epoch_number, 'test_accuracy'], round_value)}\n" \
           f"validation_accuracy: {round(model_eval_df.loc[epoch_number, 'validation_accuracy'], round_value)}\n"

fig, ax1 = plt.subplots(figsize=(13,7))

x_axis = list(model_eval_df.index)
y_min = 0.0
y_max = 0.75

ax1.plot(x_axis, list(model_eval_df.train_loss), label="training_loss", color="firebrick", dashes=[6,2])
ax1.plot(x_axis, list(model_eval_df.test_loss), label="test_loss", color="blue", dashes=[6,2])
ax1.plot(x_axis, list(model_eval_df.validation_loss), label="validation_loss", color="green", dashes=[6,2])
# ax1.plot(validation_correct_pct_per_epch.keys(), validation_correct_pct_per_epch.values(), label="ratio_correct", color="seagreen")
ax1.set_ylabel("Error", labelpad=15, fontsize=12)
ax1.set_xlabel("Epoch", labelpad=15, fontsize=12)
ax1.set_ylim(bottom=y_min, top=y_max)

ax2 = ax1.twinx()
ax2.plot(x_axis, list(model_eval_df.train_accuracy), label="train_accuacy", color="lightcoral",)
ax2.plot(x_axis, list(model_eval_df.test_accuracy), label="test_accuracy", color="lightskyblue",)
ax2.plot(x_axis, list(model_eval_df.validation_accuracy), label="validation_accuracy", color="lightgreen",)
ax2.vlines(x=epoch_number, ymin=y_min, ymax=0.4, linestyles='dotted', color='black')
ax2.set_ylabel("Accuracy", labelpad=15, fontsize=12)
ax2.set_ylim(bottom=y_min, top=y_max)

ax1.legend(loc="upper center", ncol=3, frameon=False, borderaxespad=1)
ax2.legend(loc="upper center", ncol=3, frameon=False, borderaxespad=3)

plt.text(x=200, y=0.4, s=text_box)
plt.title(f"ATC_Level1 nn {classification_type} Classifier: Training and Validation", pad=15, fontsize=16)

plt.savefig(f"/Users/leoparsons/Desktop/cheminformatics_projects/ChEMBL/ATC_level1_classifier_nn/{classification_type}_model_training.png", dpi=300)
plt.show()
