from health_sepsis.eval.metrics import classification_report_dict
def test_report_shapes():
    import numpy as np
    y = np.array([0,1,0,1,0,1,0,0,1,0])
    p = np.array([.1,.9,.2,.7,.3,.8,.4,.3,.6,.2])
    rep = classification_report_dict(y,p)
    assert "auc_roc" in rep and "auc_pr" in rep
