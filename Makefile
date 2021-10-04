

test:
	go test -v

coverage: test
	go test -coverprofile reticulum.coverprofile && go tool cover -html=reticulum.coverprofile