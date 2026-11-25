from src.stroke_prediction.api.metrics import MetricasAPI


def test_metricas_contabiliza_respostas():
    metricas = MetricasAPI()

    for status_code in [200, 201, 422, 404, 500, 503]:
        metricas.registrar(status_code)

    assert metricas.consultar() == {
        "total": 6,
        "2xx": 2,
        "4xx": 2,
        "5xx": 2,
    }


def test_metricas_protege_estado_interno():
    metricas = MetricasAPI()
    metricas.registrar(200)

    contadores = metricas.consultar()

    contadores["total"] = 999
    contadores["2xx"] = 999
    contadores["nova_chave"] = 123

    assert metricas.consultar() == {
        "total": 1,
        "2xx": 1,
        "4xx": 0,
        "5xx": 0,
    }   