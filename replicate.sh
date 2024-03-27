#----------------------------------------#
# CLASSIFICATION
#----------------------------------------#
# happy vs sad
python user_lift.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o acc8

# happy vs sad vs neutral
python user_lift.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o neutral --neutral

# hold out one user
python2 lift_leave_one_user_out.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o loov_user

# hold out emotion
python2 lift_holdout_emotion.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o holdout


#----------------------------------------#
# BOXPLOTS 
#----------------------------------------#
python2 boxplot.py -mo mo_lift_scores_acc.yaml -mu mu_lift_scores_acc.yaml -mw mw_lift_scores_acc.yaml -o fig2

# improvement over baseline
python2 error_plot.py -mo mo_lift_scores_acc.yaml -mu mu_lift_scores_acc.yaml -mw mw_lift_scores_acc.yaml -o fig3

python2 boxplot.py -mo mo_lift_scores_neutral.yaml -mu mu_lift_scores_neutral.yaml -mw mw_lift_scores_neutral.yaml -o fig4

python2 boxplot.py -mo mo_lift_scores_holdout.yaml -mu mu_lift_scores_holdout.yaml -mw mw_lift_scores_holdout.yaml -o fig5

#----------------------------------------#
# PERMUTATION TEST
#----------------------------------------#

python permute_test.py -mo mo_louo_lift_scores_loov_user.yaml -mu mu_louo_lift_scores_loov_user.yaml -mw mw_louo_lift_scores_loov_user.yaml # still error because different yaml syntax

# all features
python permute_test.py -mo mo_lift_scores_acc.yaml -mu mu_lift_scores_acc.yaml -mw mw_lift_scores_acc.yaml

# neutral
python permute_test.py -mo mo_lift_scores_neutral.yaml -mu mu_lift_scores_neutral.yaml -mw mw_lift_scores_neutral.yaml

# emotion holdout
python permute_test.py -mo mo_lift_scores_holdout.yaml -mu mu_lift_scores_holdout.yaml -mw mw_lift_scores_holdout.yaml
