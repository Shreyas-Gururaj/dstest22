#!/bin/bash
echo ${SCRIPT_MODE}
if [ "${SCRIPT_MODE}" = "regression" ]; then
    python3 ./regression/regresion_unit_price.py
elif [ "${SCRIPT_MODE}" = "train_reccomendation" ]; then
    python3 ./app/conrad_recommendation_train.py
else 
    uvicorn app.main:app --host "0.0.0.0" --port "8000"
fi
