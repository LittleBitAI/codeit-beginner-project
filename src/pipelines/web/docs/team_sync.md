# 팀 학습 실시간 동기화

각 PC가 학습을 실행하고 AWS AppSync가 설정, 상태, 로그와 결과를 팀원에게 전달한다.
실제 모델 학습을 AWS에서 실행하는 기능은 아니다.

## AWS 배포

배포는 비용이 발생할 수 있으므로 팀 AWS 관리자 승인 뒤 진행한다. AWS SAM CLI가 필요하다.

```powershell
cd src/pipelines/web/cloud
sam validate --lint
sam build
sam deploy --guided --region ap-northeast-2
```

`CognitoDomainPrefix`는 AWS 전체에서 고유한 영문 소문자 prefix를 넣는다. 배포가 끝나면
CloudFormation Output의 `PublisherPolicyArn`을 a~e가 사용하는 IAM Identity Center
permission set 또는 role에 연결한다. 이 policy는 상태와 로그를 올리는 AppSync mutation만
허용한다.

Cognito User Pool에서 a~e 계정을 관리자가 만들고 모두 `train-team` 그룹에 넣는다.
Self sign-up은 template에서 꺼져 있다. 임시 password로 처음 로그인한 뒤 새 password를
설정한다.

## 각 PC 설정

CloudFormation Output을 환경 변수로 전달한다. credential 값은 저장하지 않는다.

```text
PILL_TEAM_SYNC_ENABLED=true
PILL_TEAM_ID=<TeamId>
PILL_TEAM_APPSYNC_URL=<AppSyncUrl>
PILL_TEAM_COGNITO_USER_POOL_ID=<UserPoolId>
PILL_TEAM_COGNITO_CLIENT_ID=<UserPoolClientId>
PILL_TEAM_COGNITO_DOMAIN=<CognitoDomain>
AWS_REGION=ap-northeast-2
AWS_PROFILE=<팀 SSO profile>
```

`aws sso login --profile <팀 SSO profile>` 뒤 평소처럼 server를 실행한다. 팀 동기화가 켜진
상태에서는 AppSync에 시작 기록을 만들지 못하면 로컬 학습도 시작하지 않는다. 학습 도중
AWS 인증이나 network가 끊기면 `artifacts/web/team-sync/outbox.jsonl`에 쌓고 재연결 뒤
순서대로 전송한다.

## 이름을 지정하는 두 값은 서로 다르다

둘 다 선택 값이고 `createRun`을 IAM으로 보낼 때 쓰는 이름이지만, **켜지는 상황과 화면에
미치는 영향이 다르므로 섞어 쓰면 안 된다.**

| | `PILL_TEAM_ACTOR` | `PILL_TEAM_UNATTENDED_ACTOR` |
|---|---|---|
| 대상 | 로그인이 **불가능한** 곳 (Colab) | 로그인이 되는 PC의 밤샘 대기열 |
| 언제 쓰이나 | login token이 **아예 없을 때** | token이 있었는데 **만료돼 거절될 때** |
| 로그인 관문 | **걷어 낸다** | 건드리지 않는다 |
| 팀 활동 읽기 | 안 됨(조회는 Cognito 전용) | 로그인해 두면 됨 |

Cognito access token의 수명은 기본 한 시간인데 대기열은 앞 학습이 끝난 뒤에야 다음 항목을
시작한다. 그래서 학습이 한 시간을 넘기면 넣어 둘 때 받아 둔 token은 반드시 죽어 있다.
`PILL_TEAM_UNATTENDED_ACTOR`가 있으면 그 자리에서 SigV4로 한 번 더 보내 대기열을 이어
돌리고, 없으면 멈춰 서서 다시 로그인하라고 답한다.

로그인이 되는 PC에서 `PILL_TEAM_ACTOR`를 대신 쓰면 안 된다. 화면이 로그인 관문을 없애기
때문에 사람이 로그인을 잊은 채로 쓰다가 학습이 본인 이름 대신 그 이름으로 기록된다.

## Field을 더할 때 (중요)

AppSync subscription은 기록을 다시 읽지 않는다. **그 subscription을 깨운 mutation의
selection set에 있던 field만** 구독자에게 전달하고, 나머지는 `null`로 채운다. 그래서
`TeamRun`이나 `LogBatch`에 field를 더하거나 뺄 때는 세 곳을 함께 고친다.

1. `cloud/schema.graphql` — type 정의
2. `team_sync.py`의 `RUN_FIELDS` / `LOG_FIELDS` — mutation이 돌려받겠다고 고르는 목록
3. `frontend/src/team/cloud.ts`의 `RUN_FIELDS` / `LOG_FIELDS` — 화면이 구독하는 목록

2번이 3번보다 좁으면 팀 활동 화면이 `settings`·`summary`·`evaluation`·`lines`를 `null`로
받아 모델명과 mAP가 `-`가 되고 실시간 로그가 멈춘다. `test_web_team_sync.py`가 이를 막는다.

## 데이터와 실패 동작

- Cognito access token은 시작 요청에만 전달하며 파일이나 log에 저장하지 않는다.
- credential처럼 보이는 값과 개인 절대 경로는 AWS 전송 전에 가린다.
- DynamoDB log item은 TTL 30일이며 실제 삭제 시각은 AWS TTL 처리 시점에 따른다.
- S3 artifact는 팀이 같은 bucket 권한을 가질 때 공유 가능하다. 로컬 artifact는 자동
  업로드하지 않고 작성자 PC 전용으로 표시한다.
- heartbeat가 2분 넘게 없으면 화면에 연결 끊김 의심을 표시한다. 성공으로 추측하지 않는다.
