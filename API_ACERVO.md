# API local do Consultor

Ao abrir o `ConsultaAcervo.exe`, uma API somente leitura fica disponível em
`http://localhost:8765/api/v1`.

## Consultar registros

```text
GET /registros?termo=6801&limite=20
GET /registros?texto=francisco&livro_id=7
```

Cada item devolve livro, termo, folha, face, metadados pesquisáveis e:

- `registro_url`: JSON completo do registro;
- `foto_url`: a fotografia original em alta resolução.
- `caminho_imagem`: caminho original da fotografia no Windows;
- `nome_confirmado`: nome validado ou corrigido por uma pessoa;
- `nome_sugerido`: leitura automática ainda não confirmada;
- `nome_status`, `nome_confianca` e `nome_fonte`: rastreabilidade da leitura;
- `nome_eh_confirmado`: verdadeiro somente após confirmação/correção humana.

Uma sugestão pode ser encontrada pela pesquisa, mas nunca é apresentada pela
API como confirmação. O consumidor deve exibir `nome_sugerido` com indicação
de incerteza e usar `foto_url` para mostrar a fotografia ao lado da segunda via.

```text
GET /registros/123
GET /imagens/456
GET /health
```

O servidor escuta apenas em `127.0.0.1` por segurança. Para um sistema em
outra máquina, altere `api.host` e `api.port` no `config.yaml` do aplicativo e
libere a porta no firewall somente para a rede necessária. Os arquivos nunca
são copiados nem regravados pela API.
