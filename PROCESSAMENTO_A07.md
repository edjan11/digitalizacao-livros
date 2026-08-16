# Processamento confiável do A-07

## Uso normal

1. Abra `ConsultaAcervo.exe` e clique em **Processamento**.
2. Selecione o A-07 e clique em **Iniciar / Retomar**.
3. O OCR rápido trabalha em até quatro registros em paralelo.
4. Registros duvidosos entram na fila Qwen, que trabalha um por vez.
5. Confirme ou corrija os nomes no Revisor. Somente essa ação deixa a linha verde.

O painel pode ser fechado enquanto trabalha. **Pausar após o item atual** não
abandona o item em andamento. Ao abrir novamente, um lote que estava executando
é retomado sem repetir os itens concluídos.

## Qwen no computador atual

O OCR rápido continua recortando apenas a linha de ``que recebeu o nome de``.
Quando um item incerto chega ao Qwen estruturado, o trabalhador envia o assento
correto sem a coluna de averbações. No Xeon AVX2 desta máquina, o Qwen usa float32,
24 threads e limite de 301056 pixels; o modelo fica carregado entre os itens.
Em um único processo, após o carregamento inicial, dois testes reais levaram
29,97 s e 21,12 s e leram ``Elaine Oliveira Silva`` e ``Jeferson Nogueira
Santos``. Uma sugestão continua exigindo revisão, porque uma letra cursiva
isolada pode ser ambígua.

## Cores e estados

- verde: confirmado ou corrigido por operador;
- vermelho: exige revisão ou falhou;
- azul: pendente, processando ou informação;
- fundo neutro: sugestão automática ainda não confirmada.

## Geometria e preservação

O processamento usa uma cópia em memória. Nessa cópia, as linhas impressas são
deixadas horizontais e as bordas verticais do formulário são alinhadas. O corte
começa depois da borda esquerda do assento e termina antes da divisória de
averbações. A fotografia JPG original não recebe moldura, rotação ou recorte.

Qualquer troca da fotografia ou mudança de hash invalida os trabalhos pendentes
da versão anterior. Cada trabalho guarda registro, imagem, termo, posição,
caixa e hash para impedir que uma resposta atrasada seja salva em outro termo.

## Auditoria administrativa

O comando abaixo apenas inspeciona o estado. Acrescente `--apply` para desativar
associações Qwen incompatíveis e criar/sincronizar a fila persistente:

```powershell
.\.venv\Scripts\python.exe scripts\preparar_lote_nomes.py `
  --db "C:\ProgramData\DigitalizadorLivros\digitalizador.db" --livro A-07
```

Detecções descartadas continuam no histórico, mas deixam de participar da
pesquisa e da API como dados ativos.
## Métrica de 20 nomes e busca aproximada

O script `scripts/benchmark_qwen_20.py` processou 20 registros consecutivos do A-07 com uma única instância do modelo. O tempo médio foi 16,7 s por nome. Dezessete nomes tinham gabarito humano seguro: 8 foram exatos após remover acentos, a similaridade média de Levenshtein foi 88,85% e a distância média 2,71 caracteres. Três linhas foram marcadas sem gabarito seguro por estarem cortadas ou ambíguas.

Os arquivos `scripts/gabarito_qwen_a07_20.json` e `scripts/avaliar_benchmark_qwen.py` reproduzem a avaliação. A Consulta também faz fallback por Levenshtein quando a busca exata não encontra nome: o resultado recebe `nome_busca_similaridade` e `busca_fuzzy=true`, permanecendo sugestão e nunca confirmação automática.
Nota da medicao final entregue no executavel: similaridade media 88,88%, distancia media 2,65 caracteres, 8/17 exatos e 16,7 s por nome. Os numeros acima sao mantidos para rastrear a primeira rodada; este e o resultado com o gabarito corrigido de 6825 (Davidson Oliveira da Silva).
Atualizacao do gabarito humano: os termos 6829 (Raquila Joiane Santanna Gomes)
e 6837 (Marques Henrique dos Santos) deixaram de ficar sem gabarito seguro.
Com 19 termos avaliados, a rodada final ficou em 8/19 exatos, similaridade
media 86,06%, distancia media 3,53 caracteres e 16,5 s por nome.

## Competicao HTR no mesmo conjunto

O Kraken 7.1 foi instalado em `.venv_kraken` para nao alterar o ambiente do
aplicativo. O modelo `TraPrInq.mlmodel` foi carregado e executado nos mesmos 20
recortes de nome, com uma janela de linha deterministica e CPU. Resultado:
0/19 exatos, similaridade media 21,67% e 0,28 s por nome. O modelo reconhece
portugues manuscrito dos seculos XVI-XIX; ele confundiu principalmente o texto
impresso do formulario com a escrita cursiva moderna do A-07.

`scripts/benchmark_kraken_20.py` e `.tmp_kraken_20_result.json` guardam a
execucao reproduzivel. O recorte nao altera nenhuma fotografia original.
O BLLA (segmentador padrao) tambem foi tentado em CPU, mas nao concluiu um
recorte dentro de 2 minutos; por isso os 0,23 s acima sao apenas reconhecimento
de uma janela ja delimitada, nao o tempo de uma pagina inteira. A medicao
atualizada considera 19 nomes com gabarito.

eScriptorium + Kraken nao foi executado como servidor: nao existe instancia ou
container local ativo. O eScriptorium usa o Kraken como motor de segmentacao e
transcricao, portanto nao seria um terceiro modelo independente; com o mesmo
TraPrInq a etapa de reconhecimento teria a mesma limitacao, acrescentando a
interface e a segmentacao do servidor.

Transkribus tambem nao foi pontuado porque esta maquina nao possui conta/token
de API. A API exige um Bearer token para iniciar um trabalho. O teste pode ser
repetido depois com o modelo publico portugues (ou um modelo treinado no A-07),
sem alterar os recortes nem o gabarito.

## Orientacao para Chrome e leitor

O botao ``Abrir no Chrome (orientada)`` agora materializa uma copia temporaria
quando `rotacao_visualizacao` e 90, 180 ou 270 graus. A copia fica em
`%TEMP%\\DigitalizadorLivros\\orientadas`; o original nao e sobrescrito. Os
botoes ``Girar -90``, ``Girar +90``, ``Girar 180`` e ``Zerar rotacao`` salvam a
orientacao no banco. O leitor dedicado e o recorte enviado ao Qwen usam essa
mesma orientacao.

## Data do inicio do registro

Adicionar `data_registro` ao JSON da mesma chamada do Qwen nao exige uma segunda
passada de imagem: o custo dominante e o encoder visual. A medicao de uma
chamada de nome e de uma chamada com nome+data variou de 23,3 a 37,1 s no mesmo
registro, sem aumento consistente. Como referencia operacional, estimamos
acrescimo de 0--2 s (normalmente menos de 10%); uma segunda chamada separada
duplicaria aproximadamente o tempo. A data deve continuar como sugestao para
revisao, porque a linha manuscrita pode estar cortada ou ambigua.

Implementacao atual: a fila persistente do Qwen usa uma unica inferencia
estruturada nos registros incertos e agora guarda `nome_registrado`, `termo`,
`nome_mae` e `data_registro`. O termo continua vindo da sequencia auditada; a
resposta do Qwen nao o substitui. A data e gravada como metadado `data`, fonte
`qwen_data_registro`, status `precisa_revisao` e caixa do cabecalho, podendo ser
pesquisada sem ser apresentada como confirmada.

Teste real do 6801 apos a mudanca: uma inferencia de 88,4 s retornou os quatro
campos e a data ``25 de maio de mil novecentos``. O modelo ainda confundiu o
nome do declarante com o nome do registrado nessa leitura completa; por isso o
resultado permanece em revisao e nao substitui a leitura estreita do nome nem
a confirmacao do operador. O parser tambem recupera campos de JSON interrompido,
preservando o texto bruto para auditoria.

## Competicao de modelos locais no mesmo conjunto (20 recortes do A-07)

Em 2026-08-15 avaliamos alternativas locais ao Qwen2-VL-2B, sempre nos mesmos
recortes e com o mesmo gabarito humano. Metricas de nome (linha ``que recebeu o
nome de``):

| Motor | Exatos | Simil. media | Dist. media | Tempo/nome |
|---|---|---|---|---|
| Qwen2-VL-2B local (atual, 301056 px) | 8/19 | 86,1% | 3,53 | 16,5 s |
| Qwen2-VL-2B local (112896 px) | 6/19 | 81,1% | 4,89 | 9,7 s |
| Qwen2-VL-2B local (200704 px) | 5/19 | 75,9% | 6,74 | 12,9 s |
| Qwen2.5-VL-3B via Ollama (Q4) | 5/19 | 78,1% | 5,58 | 42,3 s |
| Qwen2.5-VL-3B via Ollama (recorte 980px) | 2/19 | 61,5% | 8,63 | 48,3 s |
| GOT-OCR 2.0 (580M, local) | 0/19 | 18,5% | 22,0 | 34,6 s |
| Kraken TraPrInq (CPU) | 0/19 | 21,7% | — | 0,28 s |

Conclusoes:

- O Qwen2-VL-2B atual continua sendo o melhor conjunto velocidade x acerto nesta
  estacao (CPU-only). O GOT-OCR 2.0 transcreve a tinta impressa do formulario
  junto com o manuscrito e fica mais lento; o 3B via Ollama e 2,5x mais lento e
  menos preciso; o Ollama redimensiona a imagem internamente, entao reduzir o
  recorte enviado nao acelera (piora).
- Reduzir `ocr.qwen_max_pixels` de 301056 para 112896 na leitura da LINHA de
  nome acelera ~40% (16,5 -> 9,7 s) com leve queda de acerto (86% -> 81%). Na
  leitura do ASSENTO inteiro (nome+termo+mae+data) o ganho cai para ~23%
  (48,5 -> 37,1 s) e o modelo perde campos como termo e data em varios itens;
  por isso o padrao permanece 301056 px.
- A origem da lentidao e o encoder visual do modelo rodando em float32 na CPU
  (sem GPU disponivel). Nenhum motor local testado supera o atual.

Reprodutibilidade: `scripts/benchmark_got_20.py`, `scripts/benchmark_ollama_20.py`
e `scripts/medir_pixels_qwen.py` / `scripts/medir_assento_qwen.py` com os
arquivos `.tmp_*_result.json` correspondentes.
