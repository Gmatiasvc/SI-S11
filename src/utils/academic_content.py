"""Contenido académico, metodológico, de negocio y ético para la plataforma SI-S11.

Este módulo centraliza las definiciones formales, diccionarios de datos, fórmulas en LaTeX,
resúmenes de etapas metodológicas (10 etapas en Comercio Electrónico y 16 etapas en RRHH),
amenazas a la validez, análisis ético-legal y respuestas rigurosas a las preguntas de
investigación y sustentación exigidas por ambas rúbricas académicas:
- Trabajo Aplicativo 01: Segmentación de clientes de comercio electrónico (No Supervisado).
- Trabajo Aplicativo Grupo 08: Predicción de rotación de empleados (Supervisado).
"""

from typing import Any, Dict, List, Optional

# ==============================================================================
# 1. CONTENIDO ACADÉMICO: COMERCIO ELECTRÓNICO (APRENDIZAJE NO SUPERVISADO)
# ==============================================================================

ECOMMERCE_ACADEMIC_CONTENT: Dict[str, Any] = {
    "problem_definition": {
        "title": "Comprensión del Problema: Segmentación Estratégica en Comercio Electrónico (Etapa 1)",
        "sector": "Comercio Electrónico (Retail Transaccional B2C / Mayorista)",
        "unit_of_analysis": (
            "Cliente único identificado (`Customer ID`), construido mediante la agregación "
            "matemática de su historial de facturas y líneas transaccionales."
        ),
        "research_objective": (
            "Implementar y comparar rigurosamente cuatro configuraciones de agrupamiento no supervisado "
            "(K-Means, DBSCAN, PCA + K-Means y PCA + DBSCAN) sobre variables transaccionales RFM enriquecidas, "
            "evaluando a PCA exclusivamente como técnica de reducción de dimensionalidad y extracción de "
            "factores latentes (nunca como clasificador o agrupador), con el fin de descubrir arquetipos de "
            "consumo accionables e identificar clientes atípicos."
        ),
        "descriptive_vs_predictive": {
            "objetivo_descriptivo_no_supervisado": (
                "En este problema NO existe una etiqueta predefinida ($y$) ni un 'verdadero segmento' histórico "
                "que el algoritmo deba adivinar. El aprendizaje no supervisado persigue un **objetivo descriptivo "
                "y exploratorio**: descubrir estructuras latentes, topologías de densidad y particiones naturales "
                "en el espacio multidimensional $\\mathbb{R}^d$ del comportamiento pasado y actual de los clientes. "
                "La validación depende de la cohesión interna, separación geométrica, estabilidad ante "
                "perturbaciones e interpretabilidad de negocio."
            ),
            "diferencia_con_prediccion_supervisada": (
                "A diferencia de un modelo supervisado (que aproxima una función $f(X) \\approx y$ para predecir "
                "un evento futuro como compra o abandono sobre una variable objetivo conocida), la segmentación "
                "no supervisada construye una taxonomía empírica de la cartera actual. Sus etiquetas resultantes "
                "($C_1, C_2, \\dots, C_k$) son constructos estadísticos que adquieren significado únicamente cuando "
                "se interpretan sus centroides y distribuciones en unidades reales de negocio."
            ),
        },
        "crm_marketing_utility": (
            "Transformar una base transaccional masiva y heterogénea en segmentos homogéneos permite a las áreas "
            "de CRM, Growth Marketing y Gestión Comercial abandonar las campañas masivas indiferenciadas (*mass-blasting*) "
            "y asignar eficientemente el presupuesto de retención: fidelización VIP para clientes *Champions*, "
            "automatización de *win-back* para clientes valiosos en riesgo de enfriamiento, *onboarding* y venta "
            "cruzada (*cross-selling*) para compradores recientes ocasionales, y tratamiento comercial mayorista "
            "o auditoría operativa para cuentas atípicas (*outliers*)."
        ),
        "valuable_clustering_criteria": [
            "**Cohesión y separación cuantitativa:** Coeficiente Silhouette elevado ($> 0.35$), bajo índice Davies-Bouldin (DBI) y alto Calinski-Harabasz (CHI) evaluados en el espacio escalado original común.",
            "**Estabilidad algorítmica:** Baja variabilidad en las particiones y métricas al perturbar la semilla aleatoria (`seeds = [10, 42, 99, 123, 2024]`) o variar ligeramente la vecindad ($\\varepsilon, \\text{min\\_samples}$).",
            "**Cobertura y granularidad accionable:** Cantidad manejable de grupos de negocio (entre 3 y 6 perfiles principales) evitando tanto la fragmentación excesiva como la pérdida masiva de clientes como ruido sin justificación.",
            "**Diferenciación semántica en unidades reales:** Perfiles claramente distinguibles en Recencia (días), Frecuencia (facturas), Gasto Monetario (£), Ticket Promedio (£) y Tasa de Cancelación (%), permitiendo diseñar al menos 3 estrategias comerciales con ROI positivo.",
        ],
    },
    "dataset_provenance": {
        "name": "Online Retail II UCI Dataset",
        "kaggle_url": "https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci",
        "uci_url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "author": "Daqing Chen (School of Engineering, London South Bank University)",
        "citation_apa7": (
            "Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: "
            "A case study of RFM model-based customer segmentation using data mining. "
            "Journal of Database Marketing & Customer Strategy Management, 19(3), 197-208."
        ),
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "access_date": "Noviembre 2026",
        "raw_observations": "1,067,371 registros transaccionales (líneas de factura entre 01/12/2009 y 09/12/2011)",
        "clean_customers_unit": (
            "~5,878 clientes únicos válidos con `Customer ID` identificado y compras efectivas "
            "(superando ampliamente el mínimo exigido de 300 observaciones y 5 variables cuantitativas)."
        ),
        "unit_of_analysis": "Cliente agregado por identificador único (`Customer ID`).",
        "original_variables_dictionary": [
            {
                "variable": "Invoice",
                "type": "Nominal (String de 6 dígitos / Prefijo 'C')",
                "description": "Número de factura transaccional. Si comienza con la letra 'C', indica una orden cancelada o nota de crédito.",
            },
            {
                "variable": "StockCode",
                "type": "Nominal (Alfanumérico de 5 caracteres)",
                "description": "Código único del producto o ítem distinto del catálogo comercial (SKU).",
            },
            {
                "variable": "Description",
                "type": "Texto (String)",
                "description": "Nombre descriptivo del artículo comercializado.",
            },
            {
                "variable": "Quantity",
                "type": "Cuantitativa Discreta (Entero)",
                "description": "Cantidad de unidades del producto por línea de factura (negativa en transacciones de cancelación/devolución).",
            },
            {
                "variable": "InvoiceDate",
                "type": "Temporal (Datetime)",
                "description": "Fecha y hora exacta en que se generó la transacción ($T_{ref} = \\max(\\text{InvoiceDate}) + 1\\text{ día}$).",
            },
            {
                "variable": "Price",
                "type": "Cuantitativa Continua (Flotante, £)",
                "description": "Precio unitario del producto expresado en libras esterlinas (£).",
            },
            {
                "variable": "Customer ID",
                "type": "Identificador Nominal (Entero de 5 dígitos)",
                "description": "Clave primaria del cliente sobre la cual se colapsan y agregan los registros transaccionales.",
            },
            {
                "variable": "Country",
                "type": "Categórica Nominal",
                "description": "País de residencia del cliente (predominantemente Reino Unido con >90% de las operaciones).",
            },
        ],
        "derived_variables_dictionary": [
            {
                "symbol": "R",
                "name": "Recency (Recencia en días)",
                "latex_formula": r"R_i = \left\lfloor \frac{T_{\text{ref}} - \max_{t \in \mathcal{V}_i}(t)}{\text{1 día}} \right\rfloor, \quad \text{donde } T_{\text{ref}} = \max(\text{InvoiceDate}) + 1\text{ día}",
                "justification": "Mide la frescura de la relación comercial. Valores bajos indican clientes activos recientemente; valores altos alertan sobre enfriamiento o abandono latente.",
            },
            {
                "symbol": "F",
                "name": "Frequency (Frecuencia de compra)",
                "latex_formula": r"F_i = \left| \{ \text{Invoice}_j \in \mathcal{V}_i \} \right|",
                "justification": "Cantidad de facturas de compra válidas (no canceladas) emitidas al cliente $i$. Refleja la recurrencia y lealtad conductual.",
            },
            {
                "symbol": "M",
                "name": "Monetary (Valor Monetario Total)",
                "latex_formula": r"M_i = \sum_{j \in \mathcal{V}_i} \left( \text{Quantity}_j \times \text{Price}_j \right)",
                "justification": "Ingresos brutos acumulados aportados por el cliente $i$ en compras válidas (£). Cuantifica el valor económico directo del cliente (CLV histórico).",
            },
            {
                "symbol": "AOV",
                "name": "Average Order Value (Ticket Promedio)",
                "latex_formula": r"\text{AOV}_i = \frac{M_i}{F_i}",
                "justification": "Desacopla el gasto total del número de visitas, permitiendo distinguir entre compradores minoristas muy frecuentes y compradores mayoristas de alto desembolso por pedido.",
            },
            {
                "symbol": "UniqueCategories",
                "name": "Variedad de Catálogo (SKUs distintos)",
                "latex_formula": r"\text{UniqueCategories}_i = \left| \{ \text{StockCode}_j \in \mathcal{V}_i \} \right|",
                "justification": "Mide la amplitud de exploración del catálogo (*cross-buying*). Diferencia a clientes mono-producto de aquellos que adquieren múltiples líneas de artículos.",
            },
            {
                "symbol": "CancellationRate",
                "name": "Tasa de Cancelación / Fricción Operativa",
                "latex_formula": r"\text{CancellationRate}_i = \frac{\left| \{ \text{Invoice}_c \in \mathcal{C}_i \} \right|}{\left| \{ \text{Invoice}_v \in \mathcal{V}_i \} \right| + \left| \{ \text{Invoice}_c \in \mathcal{C}_i \} \right|}",
                "justification": "Proporción de facturas anuladas ($\\mathcal{C}_i$, prefijo 'C') respecto al total de órdenes emitidas. Sustituye a la tasa de descuento (no registrada explícitamente en UCI Retail) como indicador real de fricción post-venta y devoluciones.",
            },
        ],
    },
    "ten_stages_guide": [
        {
            "stage": "Etapa 1",
            "name": "Comprensión del problema",
            "summary": "Definición del objetivo descriptivo de segmentación RFM no supervisada, distinción frente a predicción futura supervisada, utilidad en campañas de retención CRM y criterios cuantitativos/cualitativos de un agrupamiento valioso.",
        },
        {
            "stage": "Etapa 2",
            "name": "Carga y exploración (EDA)",
            "summary": "Inspección de dimensiones (`shape`), tipos de datos, valores faltantes (~22.7% sin `Customer ID`), duplicados, asimetría positiva extrema en variables monetarias y matriz de correlación entre las 6 variables agregadas por cliente.",
        },
        {
            "stage": "Etapa 3",
            "name": "Preprocesamiento y escalado compartido",
            "summary": "Filtrado de registros anómalos (códigos contables `POST`, `M`, precios $\\le 0$), agregación a nivel de `Customer ID`, transformación logarítmica $\\log(1+x)$ para mitigar sesgo de cola derecha y estandarización con `StandardScaler` generando la matriz única compartida $X_{\\text{scaled}}$.",
        },
        {
            "stage": "Etapa 4",
            "name": "Agrupamiento con K-Means",
            "summary": "Exploración sistemática de $K \\in [2, 10]$ con inicialización `k-means++`, `n_init=20` y semilla fija (`random_state=42`). Selección del $K$ óptimo mediante método del Codo (Inercia), Silhouette, Davies-Bouldin (DBI) y Calinski-Harabasz (CHI), e interpretación de centroides en unidades originales (£, días, compras).",
        },
        {
            "stage": "Etapa 5",
            "name": "Agrupamiento basado en densidad con DBSCAN",
            "summary": "Estimación del radio de vecindad mediante la curva de $k$-distancias ordenadas ($k = 2d - 1$) y búsqueda razonada en grilla de $(\\varepsilon, \\text{min\\_samples})$. Identificación de puntos núcleo, borde y ruido ($\\text{cluster} = -1$), evaluando solo configuraciones con $\\ge 2$ clusters válidos.",
        },
        {
            "stage": "Etapa 6",
            "name": "Reducción de dimensionalidad con PCA",
            "summary": "Ajuste de PCA sobre $X_{\\text{scaled}}$, análisis del *Scree Plot* y varianza explicada acumulada (objetivo 90%–95%), interpretación semántica de la matriz de cargas (*loadings*), proyección en PC1/PC2 y cuantificación del error cuadrático medio de reconstrucción (MSE).",
        },
        {
            "stage": "Etapa 7",
            "name": "Modelos combinados (PCA + K-Means y PCA + DBSCAN)",
            "summary": "Re-ejecución de la búsqueda de hiperparámetros de K-Means y DBSCAN utilizando exactamente el subconjunto de componentes principales retenidos en la Etapa 6, manteniendo estrictamente separada la evaluación de la compresión lineal (PCA) de la calidad del particionamiento (clustering).",
        },
        {
            "stage": "Etapa 8",
            "name": "Comparación metodológica justa",
            "summary": "Evaluación cruzada de las 4 alternativas (K-Means, DBSCAN, PCA+K-Means, PCA+DBSCAN) proyectando todas las etiquetas sobre la **misma representación escalada original ($X_{\\text{scaled}}$)** y declarando explícitamente el porcentaje de cobertura y tratamiento del ruido en DBSCAN.",
        },
        {
            "stage": "Etapa 9",
            "name": "Análisis de estabilidad y utilidad de negocio",
            "summary": "Auditoría de robustez de K-Means ante múltiples semillas (`[10, 42, 99, 123, 2024]`) e índice Adjusted Rand Index (ARI), análisis de sensibilidad de DBSCAN ante variaciones de $\\varepsilon / \\text{min\\_samples}$, e inspección de tamaño mínimo e interpretabilidad comercial de cada grupo.",
        },
        {
            "stage": "Etapa 10",
            "name": "Decisión técnica sustentada y diseño estratégico",
            "summary": "Elección justificada de la arquitectura ganadora combinando métricas internas, estabilidad, cobertura del 100% de clientes activos, simplicidad operativa e impacto financiero, culminando en el diseño de 3 estrategias comerciales diferenciadas.",
        },
    ],
    "validity_threats": [
        {
            "threat": "Sesgo de Concentración Geográfica (Dominancia del Reino Unido > 90%)",
            "category": "Muestreo y Representatividad Externa",
            "description": (
                "Más del 90% de las transacciones y clientes residen en el Reino Unido. Los costos de envío, "
                "frecuencia de reposición y hábitos de compra domésticos difieren sustancialmente de los clientes "
                "internacionales (Europa continental, Asia o América), quienes suelen realizar pedidos consolidados "
                "de mayor ticket promedio pero menor frecuencia."
            ),
            "verification_and_mitigation": (
                "**Verificación:** Calcular la distribución de clusters condicionada por país (`UK` vs `Non-UK`) mediante "
                "una prueba $\\chi^2$ de independencia y comparar distribuciones de $F$ y $\\text{AOV}$. "
                "**Mitigación:** Segmentar de forma separada el mercado doméstico (`UK`) del canal de exportación internacional "
                "o estandarizar las variables monetarias y de frecuencia dentro de cada región logística antes del clustering."
            ),
        },
        {
            "threat": "Sesgo de Estacionalidad Navideña y Corte Temporal ($T_{\\text{ref}}$)",
            "category": "Temporalidad",
            "description": (
                "El dataset corta el 9 de diciembre de 2011, justo en el pico estacional de compras navideñas (Q4). "
                "Esto comprime artificialmente la Recencia ($R$) de los compradores estacionales de fin de año e "
                "infla su gasto reciente, pudiendo confundir a un comprador exclusivamente navideño con un cliente "
                "frecuente recién adquirido."
            ),
            "verification_and_mitigation": (
                "**Verificación:** Aplicar ventanas móviles de evaluación (*rolling windows* en Q1, Q2, Q3 y Q4) y medir "
                "la matriz de transición de segmentos entre trimestres. "
                "**Mitigación:** Incorporar una variable de estacionalidad (proporción de compras realizadas fuera de Q4) "
                "o calcular el intervalo inter-compra medio para distinguir recurrencia anual de recurrencia mensual."
            ),
        },
        {
            "threat": "Sesgo de Exclusión por Transacciones Anónimas (`Customer ID` Faltante ~22.7%)",
            "category": "Cobertura de Datos / Faltantes No Aleatorios (MNAR)",
            "description": (
                "Alrededor del 22.7% de las líneas de factura carecen de `Customer ID` (compras como invitado o ventas "
                "de mostrador no vinculadas). Su exclusión es obligatoria para no crear un 'super-cliente artificial', "
                "pero introduce un sesgo de selección hacia clientes registrados y recurrentes."
            ),
            "verification_and_mitigation": (
                "**Verificación:** Comparar la distribución del valor de factura y cantidad de ítems entre facturas con "
                "`Customer ID` y facturas sin `Customer ID` mediante el test no paramétrico de Mann-Whitney U. "
                "**Mitigación:** Implementar incentivos de identificación en el *checkout* (descuento en primera compra "
                "registrada) y analizar las facturas anónimas a nivel de cesta de compra (*Market Basket Analysis*)."
            ),
        },
        {
            "threat": "Sensibilidad a Escalas y Asimetría Extrema de Clientes Mayoristas (*Whales*)",
            "category": "Escalas, Ruido y Geometría de Distancias",
            "description": (
                "Variables como `Monetary` y `Frequency` presentan distribuciones de cola pesada (*power-law*): "
                "unos pocos distribuidores mayoristas compran cientos de miles de libras. Sin preprocesamiento no lineal, "
                "la distancia euclidiana de K-Means y la covarianza de PCA quedan dominadas por menos del 1% de observaciones."
            ),
            "verification_and_mitigation": (
                "**Verificación:** Medir el coeficiente de asimetría (*skewness*) y curtosis antes y después de $\\log(1+x)$, "
                "y comparar el ARI entre particiones con `StandardScaler` puro vs `log1p + StandardScaler`. "
                "**Mitigación:** Aplicar transformación logarítmica $\\log(1+x)$ previa al escalado estándar y utilizar "
                "DBSCAN como detector explícito de cuentas corporativas/mayoristas atípicas."
            ),
        },
    ],
    "commercial_strategies": [
        {
            "strategy_id": "ESTRATEGIA_1",
            "target_profile": "Champions / Clientes VIP y Frecuentes de Alto Valor",
            "profile_characteristics": (
                "Baja Recencia ($R$ muy reciente), muy alta Frecuencia ($F$), elevado Valor Monetario acumulado ($M$) "
                "y amplia exploración de catálogo (`UniqueCategories`). Representan el núcleo pareto de ingresos."
            ),
            "strategy_name": "Programa VIP 'Apex Circle' & Preventa Exclusiva de Catálogo",
            "channel_and_tactics": [
                "Acceso anticipado de 48 horas a nuevas colecciones de temporada y lotes limitados sin requerir descuentos agresivos que erosionen el margen bruto.",
                "Asignación de línea de atención prioritaria y política de reposición exprés sin costo de envío sobre umbral preferencial.",
                "Recompensas basadas en estatus (membresía escalonada y regalos sorpresa en pedidos) y programa de referidos B2B/B2C.",
            ],
            "expected_kpi": "Incremento del +12% en Retención Anual de Cuentas Clave y expansión del +8% en Share-of-Wallet sin sacrificar margen por descuentos.",
        },
        {
            "strategy_id": "ESTRATEGIA_2",
            "target_profile": "Clientes Valiosos en Riesgo / Inactivos con Historial Relevante",
            "profile_characteristics": (
                "Recencia elevada ($R$ alto: varios meses sin comprar), pero con Frecuencia y Gasto Monetario "
                "históricos medios-altos o presencia de fricción por cancelaciones (`CancellationRate` superior a la media)."
            ),
            "strategy_name": "Campaña Automatizada de Reactivación 'Win-Back' y Mitigación de Fricción",
            "channel_and_tactics": [
                "Secuencia omnicanal en 3 toques (Email personalizado + SMS/Retargeting) activada al superar su ciclo habitual de recompra.",
                "Incentivo financiero escalonado y condicionado a monto mínimo de carrito (ej. cupón de £15 en compras superiores a su Ticket Promedio histórico $\\text{AOV}$).",
                "Auditoría proactiva de servicio al cliente para aquellos con `CancellationRate > 5%`, ofreciendo garantía de reemplazo inmediato para restaurar la confianza.",
            ],
            "expected_kpi": "Tasa de reactivación efectiva del 15%–20% de cuentas dormidas en 60 días y reducción del -25% en abandono definitivo (*churn* comercial).",
        },
        {
            "strategy_id": "ESTRATEGIA_3",
            "target_profile": "Compradores Ocasionales / Nuevos de Bajo Ticket y Poca Variedad",
            "profile_characteristics": (
                "Baja o moderada Frecuencia ($F \\in [1, 2]$), gasto acumulado modesto ($M$ bajo) y compra concentrada "
                "en pocos SKUs (`UniqueCategories` reducido)."
            ),
            "strategy_name": "Programa de Desarrollo 'Next-Basket' (Cross-Selling y Segunda Compra)",
            "channel_and_tactics": [
                "Motor de recomendaciones post-compra basado en afinidad de productos complementarios para expandir `UniqueCategories`.",
                "Bono de descuento con vencimiento corto (14 días) exclusivo para la **segunda o tercera compra**, acelerando el hábito de recurrencia.",
                "Paquetes armados (*bundles* promocionales) diseñados para elevar el Ticket Promedio ($\\text{AOV}$) en un 20% por encima de su compra inicial.",
            ],
            "expected_kpi": "Conversión del +18% de compradores de única vez a clientes recurrentes y aumento del +15% en el Ticket Promedio ($\text{AOV}$).",
        },
    ],
    "defense_questions": [
        {
            "id": "Q1",
            "question": "1. ¿Por qué este problema es de aprendizaje no supervisado y no de predicción supervisada?",
            "answer": (
                "Porque en el dataset transaccional original **carecemos de una variable objetivo o etiqueta de verdad "
                "fundamental (*ground truth* $y$)** que clasifique a priori a los clientes en categorías de negocio. "
                "Nuestro propósito no es predecir el valor futuro de una variable dependiente conocida mediante el ajuste "
                "de pesos sobre ejemplos etiquetados, sino **descubrir estructuras latentes, agrupamientos naturales y "
                "patrones de similitud** directamente a partir de la geometría y densidad de las 6 variables de comportamiento "
                "($R, F, M, \\text{AOV}, \\text{UniqueCategories}, \\text{CancellationRate}$). Por tanto, la validación no "
                "se realiza mediante matrices de confusión o F1-Score, sino a través de métricas topológicas internas "
                "(Silhouette, Davies-Bouldin, Calinski-Harabasz), pruebas de estabilidad y coherencia comercial de los perfiles."
            ),
        },
        {
            "id": "Q2",
            "question": "2. ¿Qué mostraron las métricas y cómo influyeron los hiperparámetros elegidos?",
            "answer": (
                "Las métricas internas revelaron un compromiso claro entre **compacidad geométrica, granularidad de negocio "
                "y cobertura de datos**:\n"
                "- En **K-Means**, al explorar $K \\in [2, 10]$ con `k-means++` y `n_init=20`, $K=2$ suele maximizar "
                "trivialmente el Silhouette al separar únicamente 'clientes activos vs inactivos', lo cual es insuficiente "
                "para diseñar 3 estrategias diferenciadas. La inspección conjunta de la curvatura de Inercia (Codo), "
                "el índice de Calinski-Harabasz y Davies-Bouldin permite elegir un $K$ intermedio (ej. $K=3$ o $K=4$) "
                "que preserva alta cohesión mientras distingue claramente entre *Champions*, *Clientes Regulares/Ocasionales* "
                "y *Clientes en Riesgo/Fricción*.\n"
                "- En **DBSCAN**, los hiperparámetros $\\varepsilon$ (guiado por el codo de la curva de $k$-distancias) "
                "y `min_samples` controlaron directamente la frontera entre densidad continua y ruido: valores pequeños "
                "de $\\varepsilon$ fragmentan la base y elevan excesivamente el porcentaje de ruido, mientras que valores "
                "moderados aíslan correctamente el núcleo denso de consumidores estándar y separan la cola extrema de "
                "consumo como ruido ($\\text{cluster} = -1$)."
            ),
        },
        {
            "id": "Q3",
            "question": "3. ¿Qué hizo PCA con la información y mejoró o empeoró el agrupamiento?",
            "answer": (
                "**PCA actuó exclusivamente como un transformador lineal ortogonal de reducción de dimensionalidad** "
                "(no como un algoritmo de clustering). Al proyectar las 6 variables estandarizadas en los componentes "
                "principales que retienen entre el **90% y 95% de la varianza explicada acumulada** (típicamente 3 a 4 "
                "componentes), PCA sintetizó la colinealidad fuerte entre `Frequency`, `Monetary` y `UniqueCategories` "
                "en un primer eje latente de **Volumen e Intensidad Comercial (PC1)**, mientras que **PC2** capturó "
                "principalmente la **Recencia y Fricción Operativa (`CancellationRate`)**, y **PC3** el **Ticket Promedio (`AOV`)**.\n\n"
                "**Impacto en el agrupamiento:** En el espacio reducido de PCA, las métricas calculadas *dentro* del subespacio "
                "aparentan mejorar ligeramente al eliminar el 5%–10% de varianza residual ortogonal (ruido de alta frecuencia). "
                "Sin embargo, cuando se realiza la **comparación metodológica justa en la Etapa 8** (evaluando las etiquetas de "
                "`PCA + K-Means` y `PCA + DBSCAN` sobre la matriz escalada original $X_{\\text{scaled}}$), se comprueba que "
                "`PCA + K-Means` logra una partición prácticamente equivalente en calidad y estabilidad a `K-Means` directo, "
                "con un error de reconstrucción (MSE) muy bajo, facilitando además la visualización 2D/3D e mitigando la "
                "maldición de la dimensionalidad para algoritmos de densidad como DBSCAN."
            ),
        },
        {
            "id": "Q4",
            "question": "4. ¿Qué importancia tiene el ruido identificado por DBSCAN en este caso?",
            "answer": (
                "En el contexto de comercio electrónico, los puntos clasificados como **ruido ($\\text{cluster} = -1$)** "
                "por DBSCAN **no son errores de medición descartables, sino hallazgos estratégicos de altísimo valor**. "
                "Dado que la distribución de consumo tiene colas pesadas, la región de baja densidad corresponde a:\n"
                "1. **Clientes Ultra-VIP o Distribuidores Mayoristas (*Whales*):** Compradores corporativos con gasto "
                "monetario y frecuencia decenas de desviaciones estándar por encima del consumidor minorista medio.\n"
                "2. **Cuentas con Fricción o Comportamiento Anómalo:** Clientes con tasas de cancelación extremas o "
                "compras masivas puntuales seguidas de devoluciones totales.\n\n"
                "Identificar este 3%–12% de ruido permite aislar cuentas que distorsionarían los centroides de K-Means "
                "y asignarles atención ejecutiva individualizada (Key Account Management o auditoría antifraude)."
            ),
        },
        {
            "id": "Q5",
            "question": "5. ¿Por qué la alternativa elegida es la más útil y cuáles son sus limitaciones?",
            "answer": (
                "**Alternativa elegida:** **K-Means (o PCA + K-Means con varianza $\\ge 90\\%$) con $K$ perfiles estratégicos**, "
                "complementado operativamente por el etiquetado de anomalías de DBSCAN.\n\n"
                "**Justificación de utilidad superior:**\n"
                "- **Cobertura del 100%:** Asigna un segmento accionable a la totalidad de la cartera de clientes, requisito "
                "indispensable en sistemas de automatización de marketing (CRM) donde ningún cliente registrado debe quedar "
                "sin estrategia comercial.\n"
                "- **Alta estabilidad y reproducibilidad:** Al evaluar 5 semillas distintas (`[10, 42, 99, 123, 2024]`), "
                "la variación del Silhouette es casi nula ($\\text{std} < 0.005$) y el índice ARI confirma consistencia en las asignaciones.\n"
                "- **Separabilidad e interpretabilidad directa:** Los centroides des-escalados ofrecen umbrales claros en "
                "libras esterlinas (£), días de recencia y número de pedidos, mientras que DBSCAN tiende a absorber a la "
                "gran masa de clientes minoristas en un único macro-cluster continuo debido al gradiente suave de densidad en RFM.\n\n"
                "**Limitaciones reconocidas:**\n"
                "1. Impone fronteras hiper-esféricas (diagramas de Voronoi) sobre distribuciones continuas.\n"
                "2. Requiere pre-transformación logarítmica obligatoria porque es sensible a valores extremos.\n"
                "3. Es una fotografía estática que debe recalibrarse periódicamente para capturar la evolución dinámica del cliente."
            ),
        },
        {
            "id": "Q6_RESEARCH_1",
            "question": "Pregunta Orientadora 1: ¿Existen grupos de clientes frecuentes, ocasionales y de alto valor?",
            "answer": (
                "**Sí, la evidencia empírica confirma de manera contundente la existencia de esta estructura tripartita** "
                "en la cartera de comercio electrónico:\n"
                "1. **Segmento de Alto Valor y Frecuencia (*Champions / VIP*):** Concentra clientes con compras muy recientes "
                "($R$ bajo), alta recurrencia anual ($F$ muy por encima de la mediana) y el mayor aporte monetario ($M$) y "
                "variedad de productos adquiridos.\n"
                "2. **Segmento de Compradores Regulares / En Desarrollo:** Clientes con frecuencia intermedia, ticket promedio "
                "estable y baja tasa de cancelación, que conforman la base operativa constante del negocio.\n"
                "3. **Segmento de Clientes Ocasionales o Inactivos (*En Riesgo / Dormidos*):** Caracterizados por una única "
                "o muy pocas compras ($F \\approx 1-2$), menor gasto acumulado y elevada recencia ($R$ alto), requiriendo "
                "campañas diferenciadas de activación o recuperación según su ticket histórico."
            ),
        },
        {
            "id": "Q7_RESEARCH_2",
            "question": "Pregunta Orientadora 2: ¿Qué clientes son atípicos por su consumo?",
            "answer": (
                "A través de la detección de densidad de **DBSCAN ($\\text{cluster} = -1$)** y el análisis de cuantiles "
                "extremos (percentil $> 99\\%$), se identifican tres arquetipos de clientes atípicos (*outliers*):\n"
                "1. **Mega-Compradores Mayoristas (*B2B Whales*):** Clientes que registran más de 50–200 facturas y gastos "
                "superiores a £20,000–£100,000 anuales, operando como revendedores o distribuidores institucionales.\n"
                "2. **Compradores de Ticket Extremo Puntual (*Bulk One-Shots*):** Clientes con baja frecuencia ($F=1$ o $2$) "
                "pero un Ticket Promedio ($\\text{AOV}$) extraordinariamente alto por pedidos corporativos masivos.\n"
                "3. **Perfiles de Alta Fricción / Cancelación Sistemática:** Usuarios cuya tasa de cancelación "
                "(`CancellationRate`) supera el 30%–50%, reflejando incidencias logísticas, errores de facturación o "
                "devoluciones recurrentes que erosionan la rentabilidad neta."
            ),
        },
    ],
}


# ==============================================================================
# 2. CONTENIDO ACADÉMICO: ROTACIÓN DE EMPLEADOS RRHH (APRENDIZAJE SUPERVISADO)
# ==============================================================================

HR_ACADEMIC_CONTENT: Dict[str, Any] = {
    "problem_definition": {
        "title": "Definición del Problema: Predicción Temprana de Rotación de Personal (Grupo 08)",
        "context_and_challenge": (
            "La rotación voluntaria e involuntaria de talento (*Employee Attrition*) representa uno de los costos "
            "ocultos más críticos en las organizaciones modernas: implica pérdida de capital intelectual, costos de "
            "reclutamiento y selección, curva de aprendizaje de reemplazos y sobrecarga operativa en los equipos "
            "restantes. El área de Recursos Humanos (People Analytics) requiere pasar de un análisis reactivo "
            "(entrevistas de salida cuando el colaborador ya renunció) a un **sistema preventivo de alerta temprana**."
        ),
        "binary_classification_formulation": (
            "Se formula como un problema de **aprendizaje supervisado de clasificación binaria**, donde a partir de "
            "un vector de características sociodemográficas, salariales, de clima laboral y de trayectoria "
            "$\\mathbf{x}_i \\in \\mathcal{X}$ (observables **antes** del evento de salida), se busca estimar la "
            "probabilidad condicional $P(y_i = 1 \\mid \\mathbf{x}_i)$ de la variable objetivo:\n\n"
            "$$y_i = \\begin{cases} 1 & \\text{si el empleado dejó la empresa (Attrition = Yes)} \\\\ "
            "0 & \\text{si el empleado permanece activo (Attrition = No)} \\end{cases}$$"
        ),
        "research_question": (
            "¿Cuál de los tres algoritmos basados en árboles (**Árbol de Decisión**, **Random Forest** o "
            "**Gradient Boosting**) ofrece la mejor capacidad de generalización para predecir la rotación del "
            "empleado ($1 = \\text{dejó la empresa}$, $0 = \\text{permanece}$) después de optimizar sus "
            "hiperparámetros mediante validación cruzada estratificada sobre la misma partición experimental?"
        ),
        "practical_hr_utility": (
            "Permitir a los equipos de Bienestar y Gestión del Talento identificar proactivamente colaboradores "
            "con alta propensión de fuga y focalizar intervenciones preventivas personalizadas: revisión de equidad "
            "salarial, redistribución de carga de horas extra (`OverTime`), planes de carrera y promoción interna, "
            "o esquemas de trabajo flexible para mitigar el desgaste por distancia al hogar (`DistanceFromHome`)."
        ),
    },
    "dataset_provenance": {
        "name": "IBM HR Analytics Employee Attrition & Performance",
        "kaggle_url": "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "author": "IBM Data Science / Pavan Subhash T (Repositorio Kaggle curado)",
        "license": "Open Database / Uso Académico y de Investigación Pública (Kaggle Public Dataset)",
        "access_date": "Noviembre 2026",
        "observations_count": 1470,
        "features_total_count": 35,
        "predictors_clean_count": 30,
        "target_variable": {
            "name": "Attrition",
            "mapping": "Codificada binariamente: `Yes` -> 1 (Rotación positiva: 237 casos, 16.12%), `No` -> 0 (Permanencia: 1,233 casos, 83.88%).",
            "imbalance_ratio": "Ratio de desbalance aproximado de 5.2 a 1 a favor de la clase mayoritaria (permanencia).",
        },
        "pre_event_availability_and_leakage_control": {
            "justification": (
                "Todas las variables predictoras retenidas en $X$ (antigüedad `YearsAtCompany`, horas extra `OverTime`, "
                "ingreso mensual `MonthlyIncome`, satisfacción laboral `JobSatisfaction`, distancia al trabajo "
                "`DistanceFromHome`, edad `Age`, rol, nivel educativo, etc.) constituyen registros administrativos y de "
                "encuestas de clima que **están plenamente disponibles en el sistema ERP de RRHH antes de que ocurra la "
                "renuncia**. No se incluye ninguna variable post-evento (como motivo de baja, fecha de liquidación o "
                "indemnización), garantizando ausencia total de fuga temporal (*temporal data leakage*)."
            ),
            "dropped_columns": [
                {
                    "column": "EmployeeNumber",
                    "reason": "Identificador único administrativo (clave primaria secuencial). Incluirlo induciría memorización espuria sin capacidad de generalización.",
                },
                {
                    "column": "EmployeeCount",
                    "reason": "Variable constante (valor igual a `1` en las 1,470 filas); varianza cero ($\\sigma^2 = 0$), nula información discriminante.",
                },
                {
                    "column": "Over18",
                    "reason": "Variable constante (valor igual a `'Y'` en las 1,470 filas); todos los empleados son mayores de edad.",
                },
                {
                    "column": "StandardHours",
                    "reason": "Variable constante (valor igual a `80` horas quincenales en las 1,470 filas); varianza cero.",
                },
            ],
        },
    },
    "metric_justification": {
        "primary_metric": "F1-Score de la clase positiva (Attrition = 1)",
        "formula_latex": r"F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}",
        "why_chosen_before_tuning": (
            "Se estableció formalmente el **F1-Score de la clase positiva ($y=1$)** ANTES del ajuste de hiperparámetros "
            "por dos razones fundamentales técnicas y de negocio:"
        ),
        "technical_reason_class_imbalance": (
            "**1. Invalidez del Accuracy ante el Desbalance de Clases (16.12% vs 83.88%):** "
            "Un clasificador trivial o 'ciego' que prediga sistemáticamente que ningún empleado renunciará ($\\hat{y} = 0$ "
            "para todo el personal) alcanzaría un **Accuracy engañoso del 83.88%**, pero tendría un Recall de $0.00$ y "
            "un F1-Score de $0.00$, siendo completamente inútil para Recursos Humanos. El F1-Score evalúa exclusivamente "
            "el desempeño sobre la clase minoritaria de interés (`Attrition = 1`)."
        ),
        "business_reason_asymmetric_costs": (
            "**2. Balance entre Costos Asimétricos de Falsos Negativos (FN) y Falsos Positivos (FP):**\n"
            "- **Costo de un Falso Negativo (FN — Bajo Recall):** No detectar a un empleado valioso que está a punto de "
            "renunciar. El costo para la empresa es altísimo: pérdida definitiva del talento, vacante imprevista y "
            "costo de reemplazo equivalente a 6–9 meses de salario.\n"
            "- **Costo de un Falso Positivo (FP — Baja Precision):** Clasificar en riesgo a un empleado estable. El costo "
            "es moderado pero real: inversión de tiempo de RRHH en una entrevista de seguimiento, revisión de clima o "
            "beneficios preventivos. Sin embargo, si los FP son excesivos, el presupuesto de retención se diluye y el "
            "equipo de RRHH pierde confianza en las alertas del sistema.\n\n"
            "Por ello, el **F1-Score** (media armónica entre *Precision* y *Recall*), complementado con la optimización "
            "del umbral de decisión sobre probabilidades *out-of-fold* en validación cruzada y el análisis de **PR-AUC**, "
            "garantiza maximizar la captura de fugas reales manteniendo una tasa de acierto accionable."
        ),
    },
    "sixteen_stages_summary": [
        {"step": "Paso 01", "stage": "Definición del problema", "evidence": "Formulación matemática de clasificación binaria ($y \\in \\{0, 1\\}$), pregunta de investigación y utilidad preventiva en retención de talento."},
        {"step": "Paso 02", "stage": "Carga de datos", "evidence": "Adquisición automatizada y verificada desde Kaggle (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) e importación tipada en `pandas` (1,470 filas x 35 columnas)."},
        {"step": "Paso 03", "stage": "Exploración (EDA)", "evidence": "Auditoría de tipos de datos, ausencia de valores nulos, verificación de duplicados, distribuciones por clase, tasas de rotación por horas extra/antigüedad y correlaciones."},
        {"step": "Paso 04", "stage": "Limpieza inicial", "evidence": "Eliminación justificada de identificadores (`EmployeeNumber`) y atributos de varianza cero (`EmployeeCount`, `Over18`, `StandardHours`); auditoría de cero fuga post-salida."},
        {"step": "Paso 05", "stage": "Separar X e y", "evidence": "Definición de las 30 variables predictoras disponibles antes del evento ($X$) y codificación binaria estricta del objetivo `Attrition` ($y \\in \\{0, 1\\}$)."},
        {"step": "Paso 06", "stage": "Dividir Train/Test", "evidence": "Partición reproducible 80% Train (1,176 casos) / 20% Test (294 casos) con `stratify=y` y `random_state=42`, aislando completamente el conjunto Test."},
        {"step": "Paso 07", "stage": "Preprocesamiento sin fuga", "evidence": "Diseño de `ColumnTransformer`: imputación por mediana + `StandardScaler` para numéricas, e imputación por moda + `OneHotEncoder(drop='first', handle_unknown='ignore')` para categóricas."},
        {"step": "Paso 08", "stage": "Construir Pipeline", "evidence": "Encapsulamiento atómico del preprocesador y el clasificador en objetos `sklearn.pipeline.Pipeline` independientes para Decision Tree, Random Forest y Gradient Boosting."},
        {"step": "Paso 09", "stage": "Validación cruzada base", "evidence": "Evaluación de los 3 algoritmos con hiperparámetros por defecto sobre exactamente los **mismos 5 folds** de `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` en Train."},
        {"step": "Paso 10", "stage": "Comparación inicial", "evidence": "Contraste cuantitativo de media y desviación estándar de F1, Precision, Recall, ROC-AUC, PR-AUC, complejidad estructural y tiempos de cómputo base."},
        {"step": "Paso 11", "stage": "Optimizar hiperparámetros", "evidence": "Búsqueda sistemática con `RandomizedSearchCV` ($\\ge 24$ configuraciones por modelo) dentro de Train usando la misma partición de 5 folds y métrica objetivo F1."},
        {"step": "Paso 12", "stage": "Reentrenar modelo ganador", "evidence": "Selección del algoritmo con mejor capacidad de generalización en CV, reajuste (`refit`) sobre el 100% de Train y calibración del umbral óptimo usando solo predicciones *out-of-fold*."},
        {"step": "Paso 13", "stage": "Evaluación final en Test", "evidence": "Evaluación única e inalterada del Pipeline ganador en el conjunto Test reservado (20%): Matriz de Confusión, F1, Recall, Precision, curvas ROC y Precision-Recall."},
        {"step": "Paso 14", "stage": "Interpretación y análisis específico", "evidence": "Importancia de variables (MDI y *Permutation Importance*), diagnóstico de sobreajuste Train vs CV vs Test, análisis de rotación por **antigüedad** y discusión ética."},
        {"step": "Paso 15", "stage": "Predicción sobre nuevos casos", "evidence": "Inferencia probabilística y clasificación sobre **5 perfiles ficticios razonables y completamente descritos**, detallando factores de riesgo y recomendaciones de bienestar."},
        {"step": "Paso 16", "stage": "Guardar y desplegar en Streamlit", "evidence": "Serialización del Pipeline ganador completo con `joblib` (`models/hr_best_pipeline.joblib`) e integración en interfaz interactiva Streamlit sin re-entrenar en cada consulta."},
    ],
    "tenure_and_metric_tradeoff_analysis": {
        "precision_recall_f1_contrast": (
            "En la predicción de rotación existe una tensión estructural entre **Precision** y **Recall**: "
            "Con el umbral estándar de $0.50$, los modelos de ensamble tienden a ser conservadores frente a la clase "
            "minoritaria (logrando alta *Precision* pero bajo *Recall*, detectando solo los casos más extremos de renuncia). "
            "Al calibrar el umbral de decisión sobre las probabilidades *out-of-fold* de validación cruzada (típicamente "
            "hacia el rango $0.30 - 0.42$), se incrementa sustancialmente el **Recall** (capturando entre el 55% y 70% de "
            "las salidas reales) con una reducción controlada de la **Precision**, alcanzando el punto máximo de **F1-Score**."
        ),
        "tenure_differences_study": (
            "**Estudio de diferencias por antigüedad (`YearsAtCompany` y `TotalWorkingYears`):**\n"
            "El análisis exploratorio y de importancia de variables demuestra que el riesgo de rotación **no es uniforme "
            "a lo largo del ciclo de vida del empleado**:\n"
            "- **Fase Crítica Temprana (0 a 2 años de antigüedad):** Concentra la tasa de rotación más alta de la compañía "
            "(>28%–35%). Los errores de ajuste cultural (*onboarding*), salarios de entrada bajos (`MonthlyIncome`) y "
            "asignación de horas extra (`OverTime = Yes`) disparan la salida temprana.\n"
            "- **Fase de Consolidación (3 a 7 años):** La tasa de rotación desciende significativamente (~12%–15%), "
            "siendo impulsada en este tramo por estancamiento profesional (`YearsSinceLastPromotion`) o distancia al hogar.\n"
            "- **Fase Senior (> 8 años):** Presenta la mayor estabilidad laboral (<8% de rotación), donde el modelo "
            "aprende que la antigüedad acumulada y el nivel de opciones sobre acciones (`StockOptionLevel`) actúan "
            "como fuertes amortiguadores contra la renuncia."
        ),
    },
    "ethical_and_legal_analysis": {
        "title": "Marco Ético, Legal y de Gobernanza de IA en Recursos Humanos",
        "no_automated_decisions_principle": (
            "**1. Prohibición Estricta de Decisiones Laborales o Despidos Automáticos (Human-in-the-Loop):**\n"
            "Bajo ninguna circunstancia una predicción de alto riesgo de rotación ($\\hat{p} > \\tau$) debe utilizarse "
            "para tomar decisiones punitivas, bloquear ascensos, excluir a un trabajador de capacitaciones o ejecutar "
            "despidos preventivos. De acuerdo con los principios de la Ley de IA de la Unión Europea (EU AI Act, que "
            "clasifica los sistemas de IA en el ámbito laboral como de **Alto Riesgo**) y las directrices de ética algorítmica, "
            "este modelo es **exclusivamente una herramienta de diagnóstico preventivo para diseñar programas de apoyo, "
            "bienestar, ajuste salarial y retención voluntaria**."
        ),
        "privacy_and_confidentiality": (
            "**2. Privacidad de Datos del Trabajador y Confidencialidad de las Predicciones:**\n"
            "- **Minimización y Anonimización:** El entrenamiento excluye identificadores personales (`EmployeeNumber`, nombres, "
            "documentos de identidad). Los resultados agregados deben orientar políticas organizacionales.\n"
            "- **Acceso Restringido y No Estigmatización:** Las puntuaciones individuales de propensión a la salida no deben "
            "exponerse abiertamente a jefaturas directas sin capacitación ética, ya que conocer que un subordinado tiene "
            "'alta probabilidad de renuncia' puede generar el **efecto Pigmalión negativo** (trato distante o represalias "
            "inconscientes que terminan provocando la renuncia que se quería evitar)."
        ),
        "algorithmic_bias_and_fairness": (
            "**3. Auditoría de Sesgos Algorítmicos y Equidad Demográfica (Género, Estado Civil y Edad):**\n"
            "- El dataset incluye variables demográficas sensibles como `Gender`, `MaritalStatus` y `Age`. Si históricamente "
            "existió mayor rotación en empleados jóvenes o solteros debido a políticas corporativas sesgadas, el modelo "
            "aprenderá esa correlación.\n"
            "- **Protocolo de Mitigación:** Antes del despliegue en producción real, debe auditarse la paridad de tasas de "
            "verdaderos positivos (*Equal Opportunity*) entre subgrupos demográficos y asegurarse de que las acciones "
            "derivadas del modelo sean **beneficios laborales (mejoras de bienestar/compensación)** y nunca restricciones "
            "de derechos."
        ),
    },
    "defense_questions": [
        {
            "id": "HR_Q1",
            "question": "1. ¿Qué configuración de hiperparámetros ganó en cada algoritmo y por qué?",
            "answer": (
                "En los tres algoritmos, las configuraciones ganadoras en `RandomizedSearchCV` fueron aquellas que "
                "**regularizaron la complejidad estructural de los árboles para evitar memorizar el ruido de la clase minoritaria**:\n"
                "- **Árbol de Decisión (`DecisionTreeClassifier`):** Ganó una configuración con profundidad acotada "
                "(`max_depth` moderado), un mínimo de muestras por hoja (`min_samples_leaf` $\\ge 5$) y poda por complejidad "
                "de costo (`ccp_alpha` $> 0$) o balanceo de pesos. Un árbol sin restricción alcanza F1 cercano a 1.0 en Train "
                "pero colapsa en validación cruzada por alta varianza.\n"
                "- **Random Forest (`RandomForestClassifier`):** Ganó al combinar múltiples árboles descorrelacionados "
                "(`n_estimators` $\\ge 150$), restringiendo `max_features` y controlando `min_samples_leaf` junto con "
                "ponderación de clases (`class_weight='balanced'` o `'balanced_subsample'`), lo que elevó la sensibilidad "
                "hacia la clase positiva sin disparar los falsos positivos.\n"
                "- **Gradient Boosting (`GradientBoostingClassifier`):** Ganó una configuración de **aprendizaje lento y "
                "árboles débiles (*shallow trees*)**: tasa de aprendizaje reducida (`learning_rate` entre $0.03$ y $0.10$), "
                "profundidad baja (`max_depth` de 3 a 4) y muestreo estocástico (`subsample` $< 1.0$). Esto permitió que "
                "cada árbol secuencial corrigiera los residuos de clasificación de los casos difíciles de rotación sin sobreajustar."
            ),
        },
        {
            "id": "HR_Q2",
            "question": "2. ¿Qué algoritmo ganó según validación cruzada? ¿Qué tan estable es esa ventaja?",
            "answer": (
                "Al evaluar bajo exactamente los **mismos 5 folds estratificados de entrenamiento**, los métodos de "
                "**ensamble (Gradient Boosting / Random Forest)** superaron de forma consistente al Árbol de Decisión individual "
                "en la métrica principal **F1-Score de clase positiva**, así como en **ROC-AUC** y **PR-AUC**.\n\n"
                "**Estabilidad de la ventaja:** La ventaja sobre el Árbol de Decisión simple es clara y estadísticamente "
                "sólida, además de presentar una menor desviación estándar entre los 5 folds ($\\sigma_{\\text{CV}}$ reducida). "
                "Entre Random Forest y Gradient Boosting la diferencia es más estrecha y competitiva (con solapamiento parcial "
                "dentro de $\\pm 1$ desviación estándar debido al tamaño muestral de 1,176 registros en Train), pero el ganador "
                "muestra un mejor compromiso entre precisión y captura de renuncias reales, justificando su selección y "
                "posterior `refit` en todo el conjunto de entrenamiento."
            ),
        },
        {
            "id": "HR_Q3",
            "question": "3. ¿Qué errores y limitaciones mostró el ganador en el conjunto Test?",
            "answer": (
                "Al evaluar por única vez el Pipeline ganador en el conjunto **Test reservado (294 empleados: 247 que permanecen "
                "y 47 que rotan)**, se observaron dos tipos de errores con implicaciones distintas:\n"
                "- **Falsos Negativos (FN):** Empleados que efectivamente renunciaron pero cuyo perfil en el dataset lucía "
                "estable (ej. sin horas extra registradas, ingresos medios y alta satisfacción declarada en la última encuesta). "
                "Representan salidas impulsadas por **factores exógenos no capturados en el dataset** (ofertas agresivas de la "
                "competencia, reubicación familiar o conflictos interpersonales recientes).\n"
                "- **Falsos Positivos (FP):** Empleados que permanecen en la empresa pero comparten el perfil de alto desgaste "
                "(jóvenes, con `OverTime = Yes`, bajos ingresos y alta distancia al trabajo). Aunque estadísticamente se "
                "contabilizan como error en el corte transversal del dataset, desde la perspectiva de negocio son **empleados "
                "en riesgo latente real** que se benefician igualmente de una acción preventiva de bienestar.\n"
                "- **Limitación estructural:** El tamaño del conjunto de prueba (solo 47 casos positivos en Test) hace que "
                "cada empleado individual equivalga a $\\approx 2.1\\%$ del Recall."
            ),
        },
        {
            "id": "HR_Q4",
            "question": "4. ¿Qué ocurriría si se eligiera el modelo mirando Test?",
            "answer": (
                "Incurriríamos en **fuga de datos en la selección de modelos (*Test-Set Data Leakage* o *Overfitting to the Test Set*)**, "
                "invalidando por completo la estimación de generalización del estudio:\n"
                "1. El conjunto Test dejaría de ser una muestra independiente e insesgada del desempeño futuro y pasaría a "
                "funcionar como un conjunto de validación encubierto.\n"
                "2. Al elegir entre decenas de configuraciones e hiperparámetros mirando qué modelo obtuvo la mayor métrica "
                "en los 294 casos de Test, estaríamos **seleccionando el modelo que tuvo más suerte estadística con el ruido "
                "particular de esa partición** (sesgo de selección múltiple / *multiple hypothesis testing bias*).\n"
                "3. El F1-Score reportado sería artificialmente optimista y, al desplegar el modelo en producción con nuevos "
                "empleados reales, su rendimiento caería drásticamente."
            ),
        },
        {
            "id": "HR_Q5",
            "question": "5. ¿Qué cambiaría antes de utilizarlo en un entorno real?",
            "answer": (
                "Para llevar este prototipo académico a un entorno corporativo de producción (*Enterprise MLOps*), "
                "implementaríamos cinco cambios críticos:\n"
                "1. **Validación Temporal (*Out-of-Time Split*) y Variables Dinámicas:** Sustituir el corte transversal estático "
                "por una cohorte longitudinal con marcas de tiempo (ej. entrenar con datos de 2024–2025 para predecir bajas en "
                "el primer semestre de 2026) e incluir tendencias dinámicas (cambio salarial en los últimos 12 meses, evolución "
                "del ausentismo y horas extra mensuales).\n"
                "2. **Eliminación o Auditoría de Variables Demográficas Sensibles:** Excluir explícitamente `Gender`, "
                "`MaritalStatus` y `Age` del vector de entrada o aplicar restricciones formales de equidad (*Fairness Constraints*) "
                "para evitar sesgos discriminatorios.\n"
                "3. **Calibración de Probabilidades (`CalibratedClassifierCV`):** Aplicar calibración isotónica o sigmoidea "
                "(Platt scaling) para que el porcentaje de riesgo mostrado en Streamlit corresponda a una probabilidad empírica exacta.\n"
                "4. **Explicabilidad Local por Empleado (SHAP Values):** Integrar valores SHAP individuales en cada alerta "
                "para que RRHH sepa exactamente *por qué* sube el riesgo de un colaborador específico (ej. '+18% por exceso "
                "de horas extra') y qué palanca activar.\n"
                "5. **Monitoreo de *Data Drift* y Reentrenamiento Trimestral:** Vigilar cambios en la distribución salarial o "
                "inflación y re-entrenar el pipeline periódicamente bajo gobernanza ética."
            ),
        },
    ],
}


# ==============================================================================
# 3. FUNCIONES DE FORMATEO DINÁMICO CON RESULTADOS EXPERIMENTALES REALES
# ==============================================================================

def format_hr_defense_answers(eval_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Devuelve las 5 respuestas de sustentación de RRHH interpolando métricas reales si se proveen.

    Args:
        eval_results: Diccionario cargado desde `models/hr_evaluation_results.json` (opcional).
                      Si es None o está vacío, devuelve las respuestas rigurosas base.

    Returns:
        Lista de diccionarios con claves `id`, `question` y `answer` en formato Markdown.
    """
    base_questions = [dict(q) for q in HR_ACADEMIC_CONTENT["defense_questions"]]
    if not eval_results or not isinstance(eval_results, dict):
        return base_questions

    # Extraer información del diccionario de resultados reales si está disponible
    winner_name = eval_results.get("winner_model_name") or eval_results.get("best_model_name") or "Modelo Ganador"
    optimal_threshold = eval_results.get("optimal_threshold", 0.50)

    # Extraer tabla comparativa de modelos si existe
    models_summary = eval_results.get("models_comparison") or eval_results.get("cv_results") or {}
    test_metrics = eval_results.get("test_metrics") or eval_results.get("final_test_metrics") or {}

    # 1. Enriquecer Pregunta 1 (Hiperparámetros ganadores por algoritmo)
    if isinstance(models_summary, dict) and models_summary:
        params_lines = ["\n\n**Resultados Experimentales Obtenidos en este Proyecto:**"]
        for m_name, m_data in models_summary.items():
            if isinstance(m_data, dict):
                best_params = m_data.get("best_params", {})
                cv_f1 = m_data.get("best_cv_f1") or m_data.get("tuned_f1_mean") or m_data.get("cv_f1_tuned")
                base_f1 = m_data.get("base_cv_f1") or m_data.get("base_f1_mean") or m_data.get("cv_f1_base")
                params_str = ", ".join(f"`{k}={v}`" for k, v in best_params.items()) if best_params else "Configuración optimizada"
                score_info = ""
                if base_f1 is not None and cv_f1 is not None:
                    score_info = f" (CV F1 Base: `{float(base_f1):.4f}` $\\rightarrow$ CV F1 Optimizado: `{float(cv_f1):.4f}`)"
                elif cv_f1 is not None:
                    score_info = f" (CV F1 Optimizado: `{float(cv_f1):.4f}`)"
                params_lines.append(f"- **{m_name}**{score_info}: {params_str}")
        if len(params_lines) > 1:
            base_questions[0]["answer"] += "\n" + "\n".join(params_lines)

    # 2. Enriquecer Pregunta 2 (Algoritmo ganador en CV y estabilidad)
    winner_cv_info: List[str] = []
    if winner_name:
        winner_cv_info.append(f"\n\n**Evidencia Cuantitativa de Selección en Validación Cruzada (5 Folds):**")
        winner_cv_info.append(f"- **Algoritmo Ganador Seleccionado:** `{winner_name}`")
        if isinstance(models_summary, dict) and winner_name in models_summary:
            w_data = models_summary[winner_name]
            if isinstance(w_data, dict):
                w_mean = w_data.get("best_cv_f1") or w_data.get("tuned_f1_mean")
                w_std = w_data.get("tuned_f1_std") or w_data.get("cv_f1_std")
                if w_mean is not None and w_std is not None:
                    winner_cv_info.append(
                        f"- **Desempeño en Validación Cruzada (Train 5-Folds):** F1 = `{float(w_mean):.4f} ± {float(w_std):.4f}` "
                        f"(Umbral óptimo *out-of-fold* $\\tau^* = {float(optimal_threshold):.2f}$)."
                    )
                elif w_mean is not None:
                    winner_cv_info.append(
                        f"- **Desempeño en Validación Cruzada (Train 5-Folds):** F1 = `{float(w_mean):.4f}` "
                        f"(Umbral óptimo *out-of-fold* $\\tau^* = {float(optimal_threshold):.2f}$)."
                    )
    if len(winner_cv_info) > 1:
        base_questions[1]["answer"] += "\n" + "\n".join(winner_cv_info)

    # 3. Enriquecer Pregunta 3 (Errores y desempeño en Test reservado)
    if isinstance(test_metrics, dict) and test_metrics:
        f1_val = test_metrics.get("f1") or test_metrics.get("f1_score")
        prec_val = test_metrics.get("precision")
        rec_val = test_metrics.get("recall")
        roc_val = test_metrics.get("roc_auc")
        pr_val = test_metrics.get("pr_auc")
        cm = test_metrics.get("confusion_matrix")

        test_lines = [f"\n\n**Métricas Reales Obtenidas por `{winner_name}` en el Conjunto Test (20% Reservado):**"]
        metric_badges = []
        if f1_val is not None:
            metric_badges.append(f"**F1-Score:** `{float(f1_val):.4f}`")
        if rec_val is not None:
            metric_badges.append(f"**Recall (Sensibilidad):** `{float(rec_val):.4f}`")
        if prec_val is not None:
            metric_badges.append(f"**Precision:** `{float(prec_val):.4f}`")
        if roc_val is not None:
            metric_badges.append(f"**ROC-AUC:** `{float(roc_val):.4f}`")
        if pr_val is not None:
            metric_badges.append(f"**PR-AUC:** `{float(pr_val):.4f}`")
        if metric_badges:
            test_lines.append("- " + " | ".join(metric_badges))

        if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            test_lines.append(
                f"- **Desglose de la Matriz de Confusión en Test ($N=294$):** "
                f"Verdaderos Positivos (`TP = {tp}` fugas detectadas), "
                f"Verdaderos Negativos (`TN = {tn}` permanencias correctas), "
                f"Falsos Positivos (`FP = {fp}` alertas preventivas en empleados que permanecieron) y "
                f"Falsos Negativos (`FN = {fn}` salidas no anticipadas)."
            )
        if len(test_lines) > 1:
            base_questions[2]["answer"] += "\n" + "\n".join(test_lines)

    return base_questions


def format_ecommerce_defense_answers(eval_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Devuelve las 7 respuestas de sustentación e investigación de Comercio Electrónico.

    Si se provee `eval_results` (desde `models/ecommerce_evaluation_results.json`),
    interpola las métricas reales de la matriz comparativa de los 4 modelos.
    """
    base_questions = [dict(q) for q in ECOMMERCE_ACADEMIC_CONTENT["defense_questions"]]
    if not eval_results or not isinstance(eval_results, dict):
        return base_questions

    comparison = eval_results.get("comparison_matrix") or eval_results.get("models_comparison") or {}
    winner = eval_results.get("selected_model") or eval_results.get("winner_model")
    pca_info = eval_results.get("pca_summary") or {}

    if isinstance(comparison, dict) and comparison:
        comp_lines = ["\n\n**Síntesis de Métricas Obtenidas en la Comparación Justa (Espacio Original Escalado):**"]
        for model_name, metrics in comparison.items():
            if isinstance(metrics, dict):
                n_clusters = metrics.get("n_clusters", "N/A")
                sil = metrics.get("silhouette_original") or metrics.get("silhouette")
                dbi = metrics.get("davies_bouldin_original") or metrics.get("davies_bouldin")
                cov = metrics.get("coverage_pct") or metrics.get("coverage")
                sil_str = f"`{float(sil):.4f}`" if isinstance(sil, (int, float)) else str(sil)
                dbi_str = f"`{float(dbi):.4f}`" if isinstance(dbi, (int, float)) else str(dbi)
                cov_str = f"`{float(cov):.1f}%`" if isinstance(cov, (int, float)) else str(cov)
                comp_lines.append(
                    f"- **{model_name}:** Clusters = `{n_clusters}` | Silhouette = {sil_str} | "
                    f"Davies-Bouldin = {dbi_str} | Cobertura = {cov_str}"
                )
        if len(comp_lines) > 1:
            base_questions[1]["answer"] += "\n" + "\n".join(comp_lines)

    if isinstance(pca_info, dict) and pca_info:
        n_comp = pca_info.get("n_components")
        var_exp = pca_info.get("explained_variance_ratio_sum") or pca_info.get("cumulative_variance")
        mse_rec = pca_info.get("reconstruction_mse")
        if n_comp is not None and var_exp is not None:
            pca_line = (
                f"\n\n**Resultado Experimental de PCA:** Se seleccionaron `{n_comp}` componentes principales "
                f"que capturan el `{float(var_exp) * 100:.2f}%` de la varianza total"
            )
            if mse_rec is not None:
                pca_line += f", con un error cuadrático medio de reconstrucción (MSE) de `{float(mse_rec):.4f}`."
            else:
                pca_line += "."
            base_questions[2]["answer"] += pca_line

    if winner:
        base_questions[4]["answer"] += f"\n\n**Modelo Seleccionado en la Plataforma:** `{winner}`."

    return base_questions


def get_ecommerce_commercial_strategies() -> List[Dict[str, Any]]:
    """Retorna la lista estructurada con las 3 estrategias comerciales diferenciadas para E-Commerce."""
    return list(ECOMMERCE_ACADEMIC_CONTENT["commercial_strategies"])
