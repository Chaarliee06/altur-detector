# Altur: detector de comportamiento conversacional

`POST /detect` clasifica al **llamante** de una llamada telefónica como humano o sintético. Usa el canal 1 para contextualizar las respuestas del canal 0. Extrae turnos del WAV con un VAD por energía y aplica el baseline de 58 variables temporales con HistGradientBoosting.

**Estado:** Fase 1 aprobada. Fase 2 validada por HTTP local; falta verificar el despliegue público. Docker queda aplazado por indicación del usuario y no bloquea esta fase. Las fases 3–6 no se han iniciado porque la segunda puerta sigue cerrada. Métricas y alcance: [RESULTS.md](RESULTS.md).

Código del equipo: [Chaarliee06/altur-detector](https://github.com/Chaarliee06/altur-detector). El repositorio conserva acceso privado.

## Resultado comprobado

En las 71 llamadas de val, con hablantes separados de train: **66/71 correctas (92.96%)**, AUC **0.9849**, EER **0.0541**, Brier **0.0453**. El endpoint reproduce las probabilidades offline dentro de `1.33e-16`. Ninguna llamada completa de val produjo abstención.

La selección del VAD se hizo con 540 configuraciones y solo correlaciones de train. `lat_med` pasó de r=0.8587 a **0.9477**. No se usaron etiquetas de val para elegir parámetros ni se entrenó el clasificador con val.

## Ejecutar el modelo entregado

Requiere Python 3.12. Para reproducir la malla, usar Linux o WSL: `phase1.py` utiliza procesos `fork`. El modelo entregado permite servir sin descargar el dataset.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python serve.py
```

El servicio escucha en `0.0.0.0:8000`; respeta `PORT` cuando lo proporciona el host. `GET /health` devuelve disponibilidad y los SHA-256 del modelo y del código de inferencia con sus dependencias. No requiere GPU ni servicios de inferencia externos.

Petición:

```json
{"audio": "<WAV en base64>", "format": "wav"}
```

Respuesta:

```json
{"is_synthetic": true, "confidence": 0.87}
```

`confidence` es la probabilidad de **la clase devuelta**, conservando el contrato del baseline adjunto. Para AUC/Brier se recupera `P(synthetic)` como `confidence` si el veredicto es `true`, o `1-confidence` si es `false`. No se redondean probabilidades. La calibración adicional corresponde a la Fase 5 y sigue pendiente.

Audio mono, duración menor de 3 segundos, silencio y menos de cuatro turnos útiles devuelven `{"is_synthetic": false, "confidence": 0.5}`. A confianza 0.5, el bool es un valor exigido por el contrato y **no constituye un veredicto humano**. WAV corrupto, frecuencia distinta de 8 kHz y formato distinto de estéreo PCM16 producen 422. Hay límites de 32 MiB y 900 segundos. El audio se procesa en memoria, sin almacenarlo ni incluirlo en errores de validación.

Ejemplo local con un WAV propio:

```bash
.venv/bin/python - <<'PY'
import base64, json, urllib.request
from pathlib import Path
payload = json.dumps({"audio": base64.b64encode(Path("llamada.wav").read_bytes()).decode()}).encode()
request = urllib.request.Request("http://127.0.0.1:8000/detect", data=payload,
                                 headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request, timeout=60).read().decode())
PY
```

## Reproducir por puertas

```bash
make all PUBLIC_URL=https://TU-SERVICIO.onrender.com
```

`make all` instala dependencias, descarga la versión oficial fijada con comprobación SHA-256, ejecuta la búsqueda de Fase 1, prueba HTTP local y verifica 15 minutos el despliegue público. Se detiene con error en la primera puerta incumplida. **No despliega ni contrata recursos automáticamente y no afirma ejecutar las fases 3–6, todavía pendientes.** Docker se omite por indicación del usuario. Sin una URL pública verificada, la Fase 2 permanece pendiente.

Para repetir solo lo que ya se puede ejecutar:

```bash
make install
make data
make phase1
make phase2-local
```

`make data` descarga aproximadamente 671 MB de audio a la máquina local, además de los metadatos del repo oficial. El ZIP de este proyecto no contiene el dataset. Si ya están `audio/`, `turns/` y `manifest.csv`, los reutiliza tras verificar el manifiesto.

Para evaluar una instancia local ya levantada:

```bash
.venv/bin/python evaluate_http.py --url http://127.0.0.1:8000
```

El evaluador corre las 71 llamadas, guarda sus predicciones localmente y reporta accuracy, AUC, EER interpolado, Brier y p50/p95. Falla si difiere del resultado offline. EER legado se conserva como columna separada para comparar con el script original.

## Despliegue en Render con Python

`render.yaml` configura Python 3.12, instala `requirements.lock`, arranca con `python serve.py` y comprueba `/health`. Usa el modelo ya entrenado que está en `artifacts/model.joblib`; no ejecuta entrenamiento ni descarga datos en Render. `.python-version` fija la rama Python 3.12 conforme a la [configuración oficial de Python en Render](https://render.com/docs/python-version).

[Crear el servicio desde el Blueprint del repositorio](https://dashboard.render.com/blueprint/new?repo=https://github.com/Chaarliee06/altur-detector)

En Render, seleccionar el repositorio privado, permitir su acceso si se solicita y aplicar el Blueprint. La configuración usa el plan gratuito y no solicita secretos de aplicación. El plan gratuito se suspende tras 15 minutos sin tráfico: abrir `/health` y completar la verificación antes del benchmark. La prueba de disponibilidad solo respalda los 900 segundos efectivamente comprobados. [Comportamiento del plan gratuito](https://render.com/docs/free).

Los despliegues automáticos por commit están desactivados para controlar cuándo se cambia el modelo que sirve al jurado. Después de actualizar el código o modelo, desplegar el commit validado y volver a comprobarlo.

Tras publicar:

```bash
make phase2-public PUBLIC_URL=https://TU-SERVICIO.onrender.com
```

La restricción de mantener el audio local impide mandar las llamadas de val al host público. La métrica se comprueba por HTTP local; el host público se comprueba con tonos generados, contrato, SHA-256 del modelo y del código, y disponibilidad durante 900 segundos. No presentar ese chequeo público como una evaluación remota de las 71 llamadas.

## Docker: aplazado

Los archivos permanecen preparados para retomarlos después. Docker no forma parte de `make all` ni de la puerta pública actual.

```bash
docker build -t altur-detector .
docker run --rm -p 8000:8000 altur-detector
```

La imagen incluye únicamente el código de inferencia, dependencias fijadas y `artifacts/model.joblib`; ejecuta con UID sin privilegios. `make phase2-docker` levanta esa imagen en el puerto local 8001 y ejecuta la evaluación completa sin montar el dataset dentro del contenedor. La presencia del Dockerfile no equivale a haber superado esa comprobación.

Respaldo previsto, si ngrok ya está instalado y autenticado en la máquina del equipo:

```bash
ngrok http 8000
```

Su uso como respaldo no sustituye la prueba de estabilidad del host.

## Qué se conservó del baseline

- Las 58 variables predictoras originales; el CSV suma 61 columnas al incluir `anon_id`, `label`, `split`.
- El mismo clasificador y sus hiperparámetros. Únicamente cambió la segmentación a la configuración validada por malla.
- El manejo nativo de NaN de HGB, idéntico al entrenamiento, sin imputación distinta al servir.
- `baseline_original/` conserva los cuatro archivos recibidos para auditoría. **No ejecutar su `train.py`: el original termina ajustando sobre train+val.** El entrenamiento de este proyecto usa exclusivamente train.

No hay identificadores de hablante disponibles para construir folds nuevos: solo se usa la separación oficial train/val y no se publican resultados de CV aleatoria como evidencia de generalización a hablantes nuevos.

## Guion provisional de 60 segundos

“Detectamos la dinámica de la conversación: cómo responde el llamante al agente, sus pausas, interrupciones y solapamientos. Primero comprobamos que esas señales pueden extraerse del WAV que recibirá el jurado: la latencia mediana obtenida por VAD correlaciona 0.948 con la referencia. Con el modelo entrenado únicamente sobre esos turnos de audio acertamos 66 de 71 llamadas de hablantes separados del entrenamiento. El endpoint reproduce exactamente esa evaluación. El siguiente paso obligatorio es demostrar la disponibilidad pública del servicio.”

No afirmar todavía robustez a todos los motores desconocidos, acierto a 30 segundos, eficacia ante sintéticos acelerados ni mejora por semántica: esas pruebas corresponden a fases posteriores. La independencia de identidad acústica es una hipótesis de diseño, no una garantía demostrada sobre el set oculto.

Datos y condiciones: [repo oficial](https://github.com/alturio/hackmty26), [release v1.0](https://github.com/alturio/hackmty26/releases/tag/v1.0). Uso exclusivo HackMTY 2026; no redistribuir el dataset ni intentar identificar participantes.
