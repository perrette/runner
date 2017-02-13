test: unittest

unittest:
	for f in tests/test*py; do python $$f; done

pytest:
	py.test tests

ci:
	gitlab-runner exec shell tests

#TODO: check how to use with docker: https://gitlab.com/gitlab-org/gitlab-ci-multi-runner/blob/master/docs/commands/README.md#gitlab-runner-exec
# gitlab-runner exec docker tests


clean:
	rm -fr build* 
	find . -name "*.pyc" -delete
	find . -name __pycache__ -type d -delete
