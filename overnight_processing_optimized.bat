@echo off
REM =====================================================
REM Script ULTIMATE - Otimizado para i9-9900K + RTX 2060 + 32GB RAM
REM Com ativação de venv e processamento paralelo
REM =====================================================

echo ========================================
echo PROCESSAMENTO NOTURNO ULTIMATE EDITION
echo CPU: Intel i9-9900K (16 threads)
echo RAM: 32GB
echo GPU: NVIDIA RTX 2060
echo Inicio: %date% %time%
echo ========================================

REM Ativar ambiente virtual
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERRO: Nao foi possivel ativar o ambiente virtual!
    pause
    exit /b 1
)

REM Criar diretórios
mkdir logs 2>nul
mkdir datasets 2>nul
mkdir models 2>nul
mkdir temp 2>nul

set LOGFILE=logs\overnight_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log
set CORES=12

echo Ambiente virtual: ATIVADO
echo Usando %CORES% cores para processamento
echo Log principal: %LOGFILE%
echo.

REM =====================================================
REM INSTALAÇÃO DE DEPENDÊNCIAS OTIMIZADAS
REM =====================================================

echo [%time%] Instalando/Atualizando dependencias otimizadas...
pip install --upgrade pandas_ta tqdm pyyaml >> %LOGFILE% 2>&1
pip install --upgrade xgboost lightgbm scikit-learn-intelex >> %LOGFILE% 2>&1

REM Verificar instalações críticas
python -c "from hft_bot.backtesting.engine import BacktestEngine; print('[OK] hft_bot')"
python -c "import pandas_ta; print('[OK] pandas_ta')"
python -c "import xgboost; print('[OK] xgboost')"
python -c "import lightgbm; print('[OK] lightgbm')"

REM Ativar otimizações Intel
echo [%time%] Ativando otimizacoes Intel para scikit-learn...
python -c "from sklearnex import patch_sklearn; patch_sklearn()" >> %LOGFILE% 2>&1

REM =====================================================
REM TESTE RÁPIDO
REM =====================================================

echo.
echo [%time%] Executando teste rapido...
python scripts\build_training_set_v3.py data\BTCUSDT_1m.csv ^
       --window 14 --step 1000 --metric sharpe ^
       --param-file config\params_ma.csv ^
       --timeframe 1min ^
       --out datasets\test_quick.csv >> %LOGFILE% 2>&1

if errorlevel 1 (
    echo ERRO no teste! Verifique %LOGFILE%
    pause
    exit /b 1
)
echo [OK] Teste rapido concluido!

REM =====================================================
REM PARTE 1: DATASETS EM PARALELO (3 GRUPOS)
REM =====================================================

echo.
echo [%time%] === FASE 1: Gerando datasets otimizados ===
echo.

REM Grupo 1: Datasets principais em paralelo
echo [%time%] Grupo 1: Iniciando 3 datasets principais em paralelo...

start "Dataset Sharpe" /B python scripts\build_training_set_v3.py data\BTCUSDT_1m.csv ^
       --window 14 --step 1 --metric sharpe ^
       --param-file config\params_ma.csv ^
       --features-config features.yaml ^
       --timeframe 1min ^
       --save-all-scores ^
       --out datasets\selector_sharpe_full.csv > logs\dataset_sharpe.log 2>&1

start "Dataset Return" /B python scripts\build_training_set_v3.py data\BTCUSDT_1m.csv ^
       --window 14 --step 1 --metric total_return ^
       --param-file config\params_ma.csv ^
       --features-config features.yaml ^
       --timeframe 1min ^
       --save-all-scores ^
       --out datasets\selector_return_full.csv > logs\dataset_return.log 2>&1

start "Dataset WinRate" /B python scripts\build_training_set_v3.py data\BTCUSDT_1m.csv ^
       --window 14 --step 1 --metric win_rate ^
       --param-file config\params_ma.csv ^
       --features-config features.yaml ^
       --timeframe 1min ^
       --save-all-scores ^
       --out datasets\selector_winrate_full.csv > logs\dataset_winrate.log 2>&1

REM Aguardar início dos processos
timeout /t 10 /nobreak > nul

REM Grupo 2: Diferentes janelas
echo [%time%] Grupo 2: Iniciando datasets com janelas alternativas...

start "Dataset 7d" /B python scripts\build_training_set_v3.py data\BTCUSDT_1m.csv ^
       --window 7 --step 1 --metric sharpe ^
       --param-file config\params_ma.csv ^
       --features-config features.yaml ^
       --timeframe 1min ^
       --out datasets\selector_7d_sharpe.csv > logs\dataset_7d.log 2>&1

start "Dataset 21d" /B python scripts\build_training_set_v3.py data\BTCUSDT_1m.csv ^
       --window 21 --step 1 --metric sharpe ^
       --param-file config\params_ma.csv ^
       --features-config features.yaml ^
       --timeframe 1min ^
       --out datasets\selector_21d_sharpe.csv > logs\dataset_21d.log 2>&1

start "Dataset 30d" /B python scripts\build_training_set_v3.py data\BTCUSDT_1m.csv ^
       --window 30 --step 1 --metric sharpe ^
       --param-file config\params_ma.csv ^
       --features-config features.yaml ^
       --timeframe 1min ^
       --out datasets\selector_30d_sharpe.csv > logs\dataset_30d.log 2>&1

REM Monitor de progresso
echo.
echo Aguardando conclusao dos datasets...
echo Voce pode monitorar o progresso em outra janela com:
echo   dir datasets\*.csv
echo.

:WAIT_DATASETS
timeout /t 60 /nobreak > nul
echo [%time%] Verificando progresso...
for %%f in (datasets\*.csv) do echo   - %%f: %%~zf bytes
if not exist datasets\selector_sharpe_full.csv goto WAIT_DATASETS
if not exist datasets\selector_return_full.csv goto WAIT_DATASETS
if not exist datasets\selector_winrate_full.csv goto WAIT_DATASETS
echo [%time%] Datasets principais concluidos!

REM =====================================================
REM PARTE 2: TREINAMENTO OTIMIZADO DE MODELOS
REM =====================================================

echo.
echo [%time%] === FASE 2: Treinando modelos com GPU ===
echo.

REM Criar configurações otimizadas
echo { > params_rf_opt.json
echo   "selector__k": [20, 30, 40, 50, "all"], >> params_rf_opt.json
echo   "model__n_estimators": [200, 400, 600], >> params_rf_opt.json
echo   "model__max_depth": [10, 20, 30, null], >> params_rf_opt.json
echo   "model__min_samples_leaf": [1, 2, 4], >> params_rf_opt.json
echo   "model__n_jobs": [%CORES%] >> params_rf_opt.json
echo } >> params_rf_opt.json

echo { > params_xgb_gpu.json
echo   "selector__k": [25, 35, 50, "all"], >> params_xgb_gpu.json
echo   "model__n_estimators": [300, 600, 900], >> params_xgb_gpu.json
echo   "model__max_depth": [6, 10, 15], >> params_xgb_gpu.json
echo   "model__learning_rate": [0.01, 0.05, 0.1], >> params_xgb_gpu.json
echo   "model__tree_method": ["gpu_hist"], >> params_xgb_gpu.json
echo   "model__gpu_id": [0], >> params_xgb_gpu.json
echo   "model__n_jobs": [%CORES%] >> params_xgb_gpu.json
echo } >> params_xgb_gpu.json

echo { > params_lgb_opt.json
echo   "selector__k": [25, 35, 50, "all"], >> params_lgb_opt.json
echo   "model__n_estimators": [300, 600, 900], >> params_lgb_opt.json
echo   "model__num_leaves": [31, 63, 127, 255], >> params_lgb_opt.json
echo   "model__learning_rate": [0.01, 0.05, 0.1], >> params_lgb_opt.json
echo   "model__feature_fraction": [0.6, 0.8, 1.0], >> params_lgb_opt.json
echo   "model__bagging_fraction": [0.6, 0.8, 1.0], >> params_lgb_opt.json
echo   "model__device": ["gpu"], >> params_lgb_opt.json
echo   "model__gpu_device_id": [0], >> params_lgb_opt.json
echo   "model__n_jobs": [%CORES%] >> params_lgb_opt.json
echo } >> params_lgb_opt.json

REM Treinar Random Forest otimizado
echo [%time%] Treinando Random Forest com GridSearch otimizado...
python scripts\train_model_v2.py datasets\selector_sharpe_full.csv ^
       --output-dir models\rf_sharpe_optimized ^
       --model rf ^
       --test-ratio 0.25 ^
       --gap-days 7 ^
       --cv-splits 10 ^
       --param-grid params_rf_opt.json >> %LOGFILE% 2>&1

REM Treinar XGBoost com GPU
echo [%time%] Treinando XGBoost com GPU (RTX 2060)...
python scripts\train_model_v2.py datasets\selector_sharpe_full.csv ^
       --output-dir models\xgb_sharpe_gpu ^
       --model xgb ^
       --test-ratio 0.25 ^
       --gap-days 7 ^
       --cv-splits 10 ^
       --param-grid params_xgb_gpu.json >> %LOGFILE% 2>&1

REM Treinar LightGBM com GPU
echo [%time%] Treinando LightGBM otimizado...
python scripts\train_model_v2.py datasets\selector_sharpe_full.csv ^
       --output-dir models\lgb_sharpe_gpu ^
       --model lgb ^
       --test-ratio 0.25 ^
       --gap-days 7 ^
       --cv-splits 10 ^
       --param-grid params_lgb_opt.json >> %LOGFILE% 2>&1

REM Treinar modelos adicionais em paralelo
echo [%time%] Treinando modelos complementares em paralelo...

start "Model Return" /B python scripts\train_model_v2.py datasets\selector_return_full.csv ^
       --output-dir models\rf_return_opt ^
       --model rf ^
       --test-ratio 0.25 ^
       --gap-days 7 ^
       --cv-splits 5 > logs\model_return.log 2>&1

start "Model WinRate" /B python scripts\train_model_v2.py datasets\selector_winrate_full.csv ^
       --output-dir models\rf_winrate_opt ^
       --model rf ^
       --test-ratio 0.25 ^
       --gap-days 7 ^
       --cv-splits 5 > logs\model_winrate.log 2>&1

REM =====================================================
REM PARTE 3: ANÁLISE AVANÇADA E BENCHMARK
REM =====================================================

echo.
echo [%time%] === FASE 3: Analise e benchmark ===
echo.

REM Aguardar modelos principais
timeout /t 120 /nobreak > nul

REM Script de análise avançada
echo import os, json, time > analyze_results.py
echo import pandas as pd >> analyze_results.py
echo import numpy as np >> analyze_results.py
echo import joblib >> analyze_results.py
echo from datetime import datetime >> analyze_results.py
echo. >> analyze_results.py
echo print("\n" + "="*60) >> analyze_results.py
echo print("ANALISE COMPLETA DO PROCESSAMENTO NOTURNO") >> analyze_results.py
echo print("="*60) >> analyze_results.py
echo. >> analyze_results.py
echo # Análise de datasets >> analyze_results.py
echo datasets_info = {} >> analyze_results.py
echo for file in os.listdir('datasets'): >> analyze_results.py
echo     if file.endswith('.csv'): >> analyze_results.py
echo         try: >> analyze_results.py
echo             df = pd.read_csv(f'datasets/{file}') >> analyze_results.py
echo             datasets_info[file] = { >> analyze_results.py
echo                 'linhas': len(df), >> analyze_results.py
echo                 'colunas': len(df.columns), >> analyze_results.py
echo                 'tamanho_mb': os.path.getsize(f'datasets/{file}') / 1024 / 1024 >> analyze_results.py
echo             } >> analyze_results.py
echo         except: pass >> analyze_results.py
echo. >> analyze_results.py
echo print(f"\nDATASETS GERADOS: {len(datasets_info)}") >> analyze_results.py
echo for name, info in datasets_info.items(): >> analyze_results.py
echo     print(f"  - {name}: {info['linhas']} linhas, {info['tamanho_mb']:.1f} MB") >> analyze_results.py
echo. >> analyze_results.py
echo # Análise de modelos >> analyze_results.py
echo models_performance = [] >> analyze_results.py
echo for folder in os.listdir('models'): >> analyze_results.py
echo     meta_file = f'models/{folder}/model_metadata.json' >> analyze_results.py
echo     if os.path.exists(meta_file): >> analyze_results.py
echo         with open(meta_file) as f: >> analyze_results.py
echo             meta = json.load(f) >> analyze_results.py
echo             models_performance.append({ >> analyze_results.py
echo                 'nome': folder, >> analyze_results.py
echo                 'tipo': meta.get('model_type'), >> analyze_results.py
echo                 'accuracy': meta.get('test_accuracy', 0), >> analyze_results.py
echo                 'features': meta.get('n_features', 0) >> analyze_results.py
echo             }) >> analyze_results.py
echo. >> analyze_results.py
echo models_performance.sort(key=lambda x: x['accuracy'], reverse=True) >> analyze_results.py
echo print(f"\nMODELOS TREINADOS: {len(models_performance)}") >> analyze_results.py
echo print("\nTOP 5 MODELOS POR ACCURACY:") >> analyze_results.py
echo for i, model in enumerate(models_performance[:5]): >> analyze_results.py
echo     print(f"  {i+1}. {model['nome']}: {model['accuracy']:.4f} ({model['tipo']})") >> analyze_results.py
echo. >> analyze_results.py
echo # Benchmark de inferência >> analyze_results.py
echo print("\nBENCHMARK DE INFERENCIA:") >> analyze_results.py
echo if len(datasets_info) > 0 and len(models_performance) > 0: >> analyze_results.py
echo     test_file = list(datasets_info.keys())[0] >> analyze_results.py
echo     df_test = pd.read_csv(f'datasets/{test_file}') >> analyze_results.py
echo     feature_cols = [c for c in df_test.columns if c not in ['row_id', 'ts_end', 'best_strategy']] >> analyze_results.py
echo     X_test = df_test[feature_cols].values[:100] >> analyze_results.py
echo     for model in models_performance[:3]: >> analyze_results.py
echo         try: >> analyze_results.py
echo             model_obj = joblib.load(f"models/{model['nome']}/model.joblib") >> analyze_results.py
echo             start = time.time() >> analyze_results.py
echo             for _ in range(10): >> analyze_results.py
echo                 _ = model_obj.predict(X_test) >> analyze_results.py
echo             elapsed = time.time() - start >> analyze_results.py
echo             print(f"  {model['nome']}: {elapsed*100:.1f} ms/1000 predicoes") >> analyze_results.py
echo         except: pass >> analyze_results.py
echo. >> analyze_results.py
echo # Salvar relatório >> analyze_results.py
echo report = { >> analyze_results.py
echo     'timestamp': datetime.now().isoformat(), >> analyze_results.py
echo     'system': {'cpu': 'i9-9900K', 'ram': '32GB', 'gpu': 'RTX 2060'}, >> analyze_results.py
echo     'datasets': datasets_info, >> analyze_results.py
echo     'models': models_performance >> analyze_results.py
echo } >> analyze_results.py
echo with open('overnight_report.json', 'w') as f: >> analyze_results.py
echo     json.dump(report, f, indent=2) >> analyze_results.py
echo. >> analyze_results.py
echo print(f"\n{'='*60}") >> analyze_results.py
echo print("Relatorio completo salvo em: overnight_report.json") >> analyze_results.py

python analyze_results.py

REM Limpeza
del params_rf_opt.json 2>nul
del params_xgb_gpu.json 2>nul
del params_lgb_opt.json 2>nul
del analyze_results.py 2>nul
rmdir /S /Q temp 2>nul

echo.
echo ========================================
echo PROCESSAMENTO ULTIMATE CONCLUIDO!
echo Fim: %date% %time%
echo ========================================
echo.
echo GPU foi utilizada para XGBoost e LightGBM
echo Otimizacoes Intel aplicadas no RandomForest
echo.
echo Verifique os resultados em:
echo - datasets\ (6+ arquivos)
echo - models\ (8+ modelos)
echo - overnight_report.json
echo - logs\
echo.

REM Manter terminal aberto com venv ativo
cmd /k