# Altur: detector de comportamiento conversacional

`POST /detect` clasifica al llamante como humano o sintético a partir de un WAV estéreo de 8 kHz: canal 0 llamante, canal 1 agente. Extrae turnos con un VAD congelado y usa HistGradientBoosting, sin GPU ni servicios de inferencia externos.

**Contrato del juez actualizado (2026-09-13):** acepta `audio_base64` y el alias anterior `audio`. Los errores de `/detect` responden HTTP 200 con abstención. El modelo de 67 señales conserva el umbral **0.5**, que ya maximiza balanced accuracy en las 71 llamadas de val: **97.30%**, accuracy **97.18%**. La verificación del contrato en la URL pública con el cliente oficial está pendiente de desplegar esta corrección; las mediciones históricas siguientes no la sustituyen.

**Fase 2 cerrada:** las 71 llamadas de val contra [Render](https://altur-detector.onrender.com) dieron **66/71 (92.96%)**, AUC **0.9849**, Brier **0.0453** y latencias externas **p50 586.69 ms / p95 839.89 ms**. Cero errores y probabilidades equivalentes al baseline local.

**Fase 3:** se probaron cinco bloques por separado y una combinación. El artefacto seleccionado añade únicamente respuesta al silencio: **69/71 (97.18%)**, AUC **0.9905**, Brier **0.0306**, verificado por el mismo endpoint en HTTP local. Tiene 67 features: 58 originales y nueve nuevas. `artifacts/baseline_model.joblib` conserva el baseline público; `artifacts/model.joblib` contiene el seleccionado. La evaluación pública registrada corresponde al baseline, y la del seleccionado corresponde a HTTP local.

**Limitación medida:** el seleccionado cae a **63.38%** al adelantar 1.5 s todos los turnos sintéticos y recalcular las señales. La mejora en val no demuestra robustez frente a motores más rápidos. Las tablas completas, definiciones y decisiones están en [RESULTS.md](RESULTS.md).

## Servir el modelo

Python 3.12, sin descargar el dataset para inferencia:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python serve.py
```

Escucha en `0.0.0.0:8000` y respeta `PORT`. `/health` devuelve los SHA-256 del modelo y del código de inferencia, el número de features, los bloques activos, `decision_threshold` y `contract_version: judge-audio-base64-v1`. Para servir otro artefacto propio:

```bash
MODEL_PATH=artifacts/baseline_model.joblib .venv/bin/python serve.py
```

Petición:

```json
{"call_id": "ejemplo", "audio_base64": "<WAV en base64>", "sample_rate": 8000, "channels": 2}
```

Respuesta:

```json
{"is_synthetic": true, "confidence": 0.87}
```

También acepta `{"audio": "<WAV en base64>", "format": "wav"}`. Ambos campos de audio son opcionales; tiene prioridad `audio_base64` si no está vacío. `call_id`, `sample_rate`, `channels`, `format` y otros metadatos se ignoran: los parámetros de audio se leen del WAV. La respuesta contiene un booleano nativo y `confidence` entre 0 y 1, como probabilidad de la clase devuelta. AUC/Brier usan `confidence` si el veredicto es sintético y `1-confidence` en caso contrario. Las probabilidades no se redondean; la calibración adicional sigue pendiente.

Audio ausente, mono, audio menor de 3 segundos, silencio, menos de cuatro turnos útiles, JSON/base64/WAV inválido, formato incompatible y cualquier excepción de `/detect` devuelven **HTTP 200** con `{"is_synthetic": false, "confidence": 0.5}`. Los límites de 32 MiB y 900 segundos también producen esa abstención. La capa exterior normaliza errores de validación, enrutamiento e inferencia antes de enviar las cabeceras. CORS permite orígenes públicos sin credenciales, incluso en las respuestas de respaldo. El audio se procesa en memoria y no se refleja en errores.

## Verificar el contrato actual

`scripts/check_endpoint.py` es una copia sin modificaciones del cliente oficial en `alturio/hackmty26`, commit `429adf76b15d1bd18e26b50f371ca4f13b0585c0`; SHA-256 del script: `593f78ceb80017e791f0f8d552ca6a7b3b6763c11beedfa1f68ab1a55c363364`.

Con `manifest.csv` y `audio/` preparados mediante `make data`, despertar primero la instancia y ejecutar el cliente del juez:

```bash
curl --fail https://altur-detector.onrender.com/health
python scripts/check_endpoint.py --url https://altur-detector.onrender.com/detect --split val --n 20
```

Esta ejecución pública es la puerta de aceptación del contrato. Las pruebas locales y el evaluador histórico son auxiliares. Para reproducir la selección del umbral y comprobar los bordes durante desarrollo:

```bash
.venv/bin/python tune_threshold.py
.venv/bin/python -m unittest -v test_judge_contract
```

La selección mantiene modelo y VAD congelados, evalúa todos los cortes distintos de las probabilidades de val y conserva 0.5 si no hay mejora estricta. `reports/threshold.json` contiene todas las métricas antes/después y `artifacts/decision_policy.json` vincula el umbral al SHA-256 del modelo. Val se usa para seleccionar; el resultado no es una estimación independiente.

## Reproducir por puertas

```bash
make all PUBLIC_URL=https://altur-detector.onrender.com
```

Instala dependencias, descarga/verifica los datos fijados, ejecuta la malla de Fase 1, verifica baseline local y público, mide los bloques de Fase 3 y ejecuta estrés. Se detiene ante una puerta fallida. Linux/WSL es necesario para la búsqueda con procesos `fork`. Docker queda aplazado por indicación del usuario.

La evaluación pública de Fase 2 compara con el baseline. Si el host ya sirve otro modelo, se detiene al detectar la diferencia; no cambia el despliegue automáticamente. Tras el cierre público ya registrado se pueden repetir los experimentos con:

```bash
make phase3 stress
```

Para evaluar directamente el baseline público:

```bash
.venv/bin/python evaluate_http.py \
  --url https://altur-detector.onrender.com \
  --allow-remote \
  --output reports/phase2_public_http.json
```

`--allow-remote` permite el envío HTTPS solicitado por el usuario a su servicio. Sin esa opción el evaluador acepta únicamente HTTP loopback. No sigue redirecciones. Los audios, predicciones por llamada y cachés siguen fuera de Git.

Para verificar el modelo seleccionado y sus bordes:

```bash
.venv/bin/python verify_selected.py
```

Para evaluarlo por HTTPS después de desplegarlo, usar como comparación offline las predicciones de la variante seleccionada:

```bash
.venv/bin/python evaluate_http.py \
  --url https://altur-detector.onrender.com \
  --allow-remote \
  --offline reports/phase3_silence_recovery_offline.csv \
  --output reports/phase3_public_http.json
```

El CSV se genera localmente al ejecutar `phase3.py`; no se distribuye. La prueba HTTP en entorno limpio se ejecutó con un servidor que tenía únicamente `requirements.lock` instalado. Para repetirla, crear ese entorno y pasarlo mediante `verify_selected.py --python /ruta/al/entorno/bin/python`.

## Qué se midió

- El VAD se seleccionó entre 540 configuraciones usando solo correlaciones de train. `lat_med` obtuvo Pearson **0.9477** en train.
- Clasificador ajustado exclusivamente con 282 llamadas de train. Las 71 de val respetan el split oficial por hablante. No se hizo CV aleatoria sin IDs de hablante.
- Bloques nuevos: recuperación tras interrupción, respuesta al silencio, consistencia, deriva y autocorrelación. Solo se aceptan aumentos estrictos de accuracy; combinar consistencia con silencio no añadió aciertos.
- Misma extracción en memoria al entrenar y servir. Leer floats de un CSV para el experimento provocó una discrepancia en un umbral de árbol; se corrigió y repitió toda la ablación.
- Estrés con modelos congelados: reducción de latencias y desplazamiento completo de turnos, manteniendo intactas las llamadas humanas. Ninguno representa una evaluación de audio real de un motor nuevo.

Val también se utiliza para seleccionar bloques: el 97.18% es exploratorio y requiere confirmación independiente. No se midieron todavía semántica, acústica, clips cortos, ruido/ganancia, calibración adicional ni demo.

## Render y modelo publicado

`render.yaml` configura runtime Python, dependencias fijadas, `python serve.py`, `/health` y el artefacto incluido. No entrena ni descarga datos en Render. Los despliegues automáticos están desactivados en ese Blueprint; aplicar un commit nuevo al servicio requiere desplegarlo y verificar `/health` y las 71 llamadas contra el artefacto correspondiente. La configuración efectiva del servicio creado desde el Dashboard puede diferir del Blueprint.

El plan gratuito puede suspender la instancia: llamar primero a `/health`. Los percentiles registrados excluyen ese chequeo previo. [Documentación de Render](https://render.com/docs/free).

Código: [Chaarliee06/altur-detector](https://github.com/Chaarliee06/altur-detector). Datos y condiciones: [repo oficial](https://github.com/alturio/hackmty26), [release v1.0](https://github.com/alturio/hackmty26/releases/tag/v1.0). Uso exclusivo HackMTY 2026; no redistribuir el dataset ni intentar identificar participantes. `baseline_original/` conserva los archivos recibidos; su script de entrenamiento original no debe ejecutarse porque ajustaba también con val.
